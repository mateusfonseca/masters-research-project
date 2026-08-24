# AWS Deployment Region
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "eu-west-1" # Defaults to EU West 1 (Ireland)
}

# EC2 Instance Sizing
variable "instance_type" {
  description = "EC2 Instance type specified in proposal"
  type        = string
  default     = "r6i.xlarge" # High-memory instance type suited for FHE processing
}

# SSH Key Pair Name
variable "key_name" {
  description = "Name of an existing AWS Key Pair for SSH access"
  type        = string
  default     = "fhe-demo-key" # Identifier registered in AWS EC2 Key Pairs
}

# Local Path to SSH Public Key
variable "public_key_path" {
  description = "Local path to the SSH public key file"
  type        = string
}

# Inbound Network Security for SSH
variable "allowed_ssh_cidr" {
  description = "CIDR block allowed for SSH access (e.g. your public IP address /32)"
  type        = string
  default     = "0.0.0.0/0" # Open to all by default; restrict to "x.x.x.x/32" for production
}

# Dynamic DNS Domain Name
variable "duckdns_domain" {
  description = "DuckDNS domain name (e.g. my-fhe-demo.duckdns.org)"
  type        = string
}