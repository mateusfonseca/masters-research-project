#!/usr/bin/env bash
set -e

# CONFIGURATION
# Check for .env file
if [ ! -f .env ]; then
  echo "ERROR: Missing required '.env' file." >&2
  echo "Please create a '.env' file in the root directory before running this script." >&2
  exit 1
fi

# Load variables from .env
set -a
source .env
set +a

# Validate required variables
MISSING_VARS=()
[ -z "$DUCKDNS_DOMAIN" ] && MISSING_VARS+=("DUCKDNS_DOMAIN")
[ -z "$DUCKDNS_TOKEN" ] && MISSING_VARS+=("DUCKDNS_TOKEN")
[ -z "$SSH_KEY_PATH" ] && MISSING_VARS+=("SSH_KEY_PATH")

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
  echo "ERROR: The following required variables are missing or empty in your '.env' file:" >&2
  for VAR in "${MISSING_VARS[@]}"; do
    echo "  - $VAR" >&2
  done
  exit 1
fi

# Local directory containing Terraform deployment scripts
TERRAFORM_DIR="./terraform"

# Credentials and paths for instance authentication
REMOTE_USER="ubuntu"
REMOTE_APP_DIR="/home/ubuntu/app"

# List of local code, configuration, and data files to synchronize to the server
FILES_TO_SYNC=(
  "src/server.py"
  "requirements.txt"
  "data/encrypted_emails_db.json"
  "data/occurrence_matrix_sparse.npz"
  "data/vocabulary.json"
)

# Required generated data files
DATA_FILES=(
  "data/encrypted_emails_db.json"
  "data/occurrence_matrix_sparse.npz"
  "data/vocabulary.json"
)

# HELPER FUNCTIONS
# Display script usage information and exit
usage() {
  echo "Usage: $0 {up|down}"
  echo "  up      Provision infrastructure, sync code files, and restart services."
  echo "  down    Teardown all AWS infrastructure created by Terraform."
  exit 1
}

# Wait for EC2 instance SSH connectivity and cloud-init completion
wait_for_bootstrap() {
  local ip=$1
  echo "Waiting for SSH connectivity on $ip..."
  # Poll until SSH connection succeeds
  until ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$REMOTE_USER@$ip" "echo SSH Ready" &>/dev/null; do
    sleep 3
  done
  echo "SSH connected. Waiting for AWS user_data/cloud-init script to finish package installation..."
  
  # Block execution until cloud-init finishes setting up venv and systemd unit files
  ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$REMOTE_USER@$ip" "cloud-init status --wait" > /dev/null
  echo "Server bootstrap complete."
}

# Generates a .desktop shortcut on the user's Desktop for Streamlit
create_desktop_shortcut() {
  local project_dir="$(pwd)"
  local desktop_path="$HOME/Desktop/StreamlitApp.desktop"

  # Detect Streamlit binary inside local venv
  local stream_bin="streamlit"
  if [ -f "$project_dir/venv/bin/streamlit" ]; then
    stream_bin="$project_dir/venv/bin/streamlit"
  elif [ -f "$project_dir/.venv/bin/streamlit" ]; then
    stream_bin="$project_dir/.venv/bin/streamlit"
  fi

  # Generate launcher with absolute paths
  cat << EOF > "$desktop_path"
		[Desktop Entry]
		Type=Application
		Name=Streamlit FHE App
		Comment=Launch Streamlit Application in Virtualenv
		Exec=/bin/bash -c "cd '$project_dir' && '$stream_bin' run app.py"
		Icon=utilities-terminal
		Terminal=false
		Categories=Application;
EOF

  chmod +x "$desktop_path"

  # Trust desktop file on GNOME desktop environments
  if command -v gio &>/dev/null; then
    gio trust "$desktop_path" 2>/dev/null || true
  fi

  echo "Desktop shortcut created/updated at: $desktop_path"
}

# MAIN WORKFLOW
# Ensure exactly one argument ({up|down}) is passed
if [ "$#" -ne 1 ]; then
  usage
fi

ACTION="$1"

case "$ACTION" in
  up)
    echo " Step 0: Ensuring Local Virtual Environment and Dependencies"
    if [ -d "venv" ]; then
      LOCAL_VENV_PATH="venv"
    elif [ -d ".venv" ]; then
      LOCAL_VENV_PATH=".venv"
    else
      echo "No local virtual environment found. Creating 'venv'..."
      python3 -m venv venv
      LOCAL_VENV_PATH="venv"
    fi

    echo "Installing local requirements into $LOCAL_VENV_PATH..."
    "$LOCAL_VENV_PATH/bin/pip" install --upgrade pip
    "$LOCAL_VENV_PATH/bin/pip" install -r requirements.txt

    echo " Step 1: Provisioning AWS Infrastructure via Terraform"
    cd "$TERRAFORM_DIR"
    terraform init
    terraform apply -auto-approve
    
    # Retrieve the public IP address of the provisioned EC2 instance
    PUBLIC_IP=$(terraform output -raw public_ip)
    cd - > /dev/null

    echo "Target Server IP: $PUBLIC_IP"

    # Wait for cloud-init to fully complete pip installs & systemd setup
    wait_for_bootstrap "$PUBLIC_IP"

    echo " Step 2: Updating DuckDNS IP Record"
    # Extract the domain prefix and update DuckDNS to point to the new EC2 Public IP
    DUCK_NAME=$(echo "$DUCKDNS_DOMAIN" | cut -d'.' -f1)
    curl -s "https://www.duckdns.org/update?domains=${DUCK_NAME}&token=${DUCKDNS_TOKEN}&ip=${PUBLIC_IP}"
    echo ""
    echo "DuckDNS record updated to point $DUCKDNS_DOMAIN -> $PUBLIC_IP"
    echo "Waiting 10 seconds for DNS propagation..."
    sleep 10

    echo " Step 3: Syncing Code Files to Remote Server"
    # Check if required dataset is missing
    if [ ! -f "data/emails.csv" ]; then
      echo "'data/emails.csv' not found. Fetching Enron Email Dataset..."
      mkdir -p data

      # Download zip only if it doesn't already exist locally
      if [ ! -f "data/emails.csv.zip" ]; then
        echo "--> Downloading dataset zip..."
        curl -L -o data/emails.csv.zip "https://www.kaggle.com/api/v1/datasets/download/wcukierski/enron-email-dataset"
      else
        echo "--> Found existing 'data/emails.csv.zip'. Skipping download."
      fi

      echo "--> Unzipping dataset..."
      unzip -q data/emails.csv.zip -d data/

      if [ ! -f "data/emails.csv" ]; then
        echo "--> ERROR: Extraction failed. 'data/emails.csv' was not created."
        exit 1
      fi
      
      # Optional: Clean up zip after successful extraction
      rm -f data/emails.csv.zip
      
      echo "Dataset successfully extracted to data/emails.csv"
    fi
    
    # Check if any required data file is missing
    MISSING_DATA=false
    for FILE in "${DATA_FILES[@]}"; do
      if [ ! -f "$FILE" ]; then
        echo "--> Missing required artifact: $FILE"
        MISSING_DATA=true
      fi
    done

    # If any file was missing, execute setup_db.py locally to generate them
    if [ "$MISSING_DATA" = true ]; then
      echo "Missing data files detected. Running setup_db.py..."
      
      # Locate local virtual environment executable
      if [ -f "venv/bin/python" ]; then
        PYTHON_BIN="venv/bin/python"
      elif [ -f ".venv/bin/python" ]; then
        PYTHON_BIN=".venv/bin/python"
      elif command -v python3 &>/dev/null; then
        PYTHON_BIN="python3"
      else
        echo "--> ERROR: Python interpreter not found."
        exit 1
      fi

      "$PYTHON_BIN" setup_db.py
    else
      echo "All required data files present."
    fi
    
    # Loop over and rsync each file in the payload list to the server
    for ITEM in "${FILES_TO_SYNC[@]}"; do
      if [ -e "$ITEM" ]; then
        echo "--> Uploading $ITEM..."
        # -R (--relative) preserves relative folder paths on the target directory
        rsync -avzR -e "ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no" "$ITEM" "$REMOTE_USER@$PUBLIC_IP:$REMOTE_APP_DIR/"
      else
        echo "--> WARNING: Local file '$ITEM' not found. Skipping."
      fi
    done

    echo " Step 4: Configuring SSL & Starting Background Services"
    # Execute remote commands on the EC2 instance via SSH heredoc
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$REMOTE_USER@$PUBLIC_IP" bash -s << EOF
      set -e
      
      # Assign variable inside remote session
      DUCKDNS_DOMAIN="$DUCKDNS_DOMAIN"

      echo "Fixing permissions..."
      sudo chown -R ubuntu:ubuntu /home/ubuntu/app

      echo "Installing python dependencies..."
      cd /home/ubuntu/app
      source venv/bin/activate
      pip install -r requirements.txt

      echo "Running Certbot to issue HTTPS SSL Certificate for domain: \$DUCKDNS_DOMAIN"
      # Checks if certificate already exists to avoid redundant re-issuances
      if [ ! -d "/etc/letsencrypt/live/\$DUCKDNS_DOMAIN" ]; then
        sudo certbot --nginx -d "\$DUCKDNS_DOMAIN" --non-interactive --agree-tos --register-unsafely-without-email --redirect --test-cert
      else
        echo "--> SSL Certificate already present."
      fi

      echo "Reloading systemd and restarting FastAPI & Nginx..."
      sudo systemctl daemon-reload
      sudo systemctl reset-failed
      sudo systemctl restart fastapi
      sudo systemctl restart nginx

      sleep 3
      
      echo "Verifying FastAPI active state..."
      if sudo systemctl is-active --quiet fastapi; then
        echo " [OK] FastAPI is ACTIVE on localhost:8000"
      else
        echo " [ERROR] FastAPI failed to start. Journal log:"
        sudo journalctl -u fastapi -n 20 --no-pager
      fi
EOF

		# Generate or refresh the local desktop shortcut
    create_desktop_shortcut

    echo " DEPLOYMENT COMPLETE!"
    echo " Secure Backend Endpoint: https://$DUCKDNS_DOMAIN"
    echo " All Data in Transit is now Encrypted via TLS v1.3"
    ;;

  down)
    echo " Destroying AWS Infrastructure"
    # Destroy all Terraform-managed AWS resources
    cd "$TERRAFORM_DIR"
    terraform destroy -auto-approve
    cd - > /dev/null
    echo "All resources destroyed."
    ;;

  *)
    usage
    ;;
esac
