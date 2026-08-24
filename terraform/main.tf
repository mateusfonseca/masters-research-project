# Terraform configuration requirements
terraform {
  # Require Terraform version 1.2 or higher
  required_version = ">= 1.2"
  
  # Specify required provider dependencies
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0" # Pin AWS provider to version 5.x
    }
  }
}

# AWS Provider configuration
provider "aws" {
  region = var.aws_region # Target deployment region passed via variables
}

# 1. Register Local Public Key in AWS Key Pair Registry
resource "aws_key_pair" "fhe_key" {
  key_name   = var.key_name
  public_key = file(var.public_key_path) # Reads the public key file from local path
}

# 2. Fetch Latest Ubuntu 24.04 LTS AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  
  # Filter for Ubuntu 24.04 (Noble Numbat) 64-bit server images
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
  
  # Ensure Hardware Virtual Machine (HVM) virtualization
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  
  owners = ["099720109477"] # Canonical's official AWS account ID
}

# 3. Security Group Configuration
resource "aws_security_group" "fhe_app_sg" {
  name        = "fhe-research-app-sg"
  description = "Security group for FHE Search Engine Benchmark Suite"

  # Inbound rule: Allow SSH traffic from restricted CIDR block
  ingress {
    description = "SSH Access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # Inbound rule: Allow HTTP traffic globally for Certbot and auto-redirection
  ingress {
    description = "HTTP Traffic (Certbot / Auto-Redirect)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Inbound rule: Allow HTTPS traffic globally for secure FHE API endpoints
  ingress {
    description = "HTTPS Traffic (Secure FHE Endpoints)"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Outbound rule: Allow all outbound network traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1" # -1 signifies all protocols
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "fhe-research-sg"
  }
}

# 4. EC2 Instance Provisioning
resource "aws_instance" "fhe_server" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.fhe_key.key_name
  vpc_security_group_ids = [aws_security_group.fhe_app_sg.id]

  # Primary storage volume configuration
  root_block_device {
    volume_size           = 30 # 30 GB disk space
    volume_type           = "gp3"
    delete_on_termination = true # Destroy volume when instance is terminated
  }

  # User Data Script: Bootstrap Dependencies, Swap Space, and Background Services on first boot
  user_data = <<-EOF
              #!/bin/bash
              set -e # Exit immediately if any command fails

              # 1. Swap File Allocation (8GB swap space to handle memory spikes during FHE operations)
              fallocate -l 8G /swapfile
              chmod 600 /swapfile
              mkswap /swapfile
              swapon /swapfile
              echo '/swapfile none swap sw 0 0' >> /etc/fstab

              # 2. Add Repositories & Dependencies (Nginx, Certbot, Python 3.14 via deadsnakes PPA)
              apt-get update -y
              apt-get install -y software-properties-common git htop build-essential rsync nginx certbot python3-certbot-nginx
              add-apt-repository -y ppa:deadsnakes/ppa
              apt-get update -y
              apt-get install -y python3.14 python3.14-venv python3.14-dev

              # 3. Environment & Workspace Setup
              mkdir -p /home/ubuntu/app
              cd /home/ubuntu/app

              # Create virtual environment explicitly using Python 3.14 and upgrade pip
              python3.14 -m venv venv
              source venv/bin/activate
              pip install --upgrade pip

              # Set ownership permissions to the default ubuntu user
              chown -R ubuntu:ubuntu /home/ubuntu/app

              # 4. Systemd Service Configuration for FastAPI
              cat << 'SERVICE' > /etc/systemd/system/fastapi.service
              [Unit]
              Description=FHE FastAPI Backend Server
              After=network.target

              [Service]
              User=ubuntu
              WorkingDirectory=/home/ubuntu/app/src
              ExecStart=/home/ubuntu/app/venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
              Restart=always
              RestartSec=3s

              [Install]
              WantedBy=multi-user.target
              SERVICE

              # 5. Default Nginx Catch-All HTTP Proxy Config
              # Set client_max_body_size globally to 100M so all HTTP & future HTTPS blocks inherit it
              sudo sed -i 's/http {/http {\n    client_max_body_size 100M;/' /etc/nginx/nginx.conf

              # Create Nginx Reverse Proxy Config for FastAPI backend
              cat << 'NGINX' > /etc/nginx/sites-available/fhe_proxy
              server {
                  listen 80 default_server;
                  listen [::]:80 default_server;
                  server_name _;

                  client_max_body_size 100M;

                  location / {
                      proxy_pass http://127.0.0.1:8000;
                      proxy_set_header Host $host;
                      proxy_set_header X-Real-IP $remote_addr;
                      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                      proxy_set_header X-Forwarded-Proto $scheme;

                      # Timeouts (300s) extended for heavy FHE computation payloads
                      proxy_read_timeout 300;
                      proxy_connect_timeout 300;
                      proxy_send_timeout 300;
                  }
              }
              NGINX

              # Enable the custom Nginx configuration and remove the default config
              ln -sf /etc/nginx/sites-available/fhe_proxy /etc/nginx/sites-enabled/
              rm -f /etc/nginx/sites-enabled/default

              # Reload systemd, enable the FastAPI service, and restart Nginx
              systemctl daemon-reload
              systemctl enable fastapi
              systemctl restart nginx
              EOF

  tags = {
    Name        = "FHE-Benchmark-Instance"
    Environment = "Research"
  }
}