# Output: Public IP address assigned to the provisioned EC2 instance
output "public_ip" {
  description = "Public IP address of the deployed EC2 instance"
  value       = aws_instance.fhe_server.public_ip
}

# Output: The secure HTTPS endpoint URL for API requests, routed through DuckDNS and Nginx
output "fastapi_api_https_url" {
  description = "HTTPS API URL for the FastAPI backend via DuckDNS/Nginx"
  value       = "https://${var.duckdns_domain}"
}

# Output: Pre-formatted terminal command for convenient SSH access to the server
output "ssh_connection_string" {
  description = "Command to SSH into the instance"
  value       = "ssh -i ~/.ssh/fhe-demo-key ubuntu@${aws_instance.fhe_server.public_ip}"
}