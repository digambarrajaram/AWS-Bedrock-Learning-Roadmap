module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "3.14.0"

  name = "corporate-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-west-2a", "us-west-2b", "us-west-2c"]
  private_subnets = ["10.0.0.0/19", "10.0.32.0/19", "10.0.64.0/19"]
  public_subnets  = ["10.0.96.0/19", "10.0.128.0/19", "10.0.160.0/19"]

  enable_nat_gateway     = true
  single_nat_gateway     = false
  enable_vpc_flow_log    = true
  enable_dns_hostnames   = true
  enable_dns_support     = true
  flow_log_cloudwatch_log_group_retention_days = 30
  flow_log_traffic_type = "ALL"
  flow_log_destination_type = "cloud-watch-logs"

  public_subnet_tags = {
    Name        = "public-subnet"
    Environment = "production"
  }

  private_subnet_tags = {
    Name        = "private-subnet"
    Environment = "production"
  }

  nat_gateway_tags = {
    Name        = "nat-gateway"
    Environment = "production"
    Terraform   = "true"
  }

  flow_log_tags = {
    Name        = "vpc-flow-log"
    Environment = "production"
    Terraform   = "true"
  }

  igw_tags = {
    Name        = "internet-gateway"
    Environment = "production"
    Terraform   = "true"
  }

  tags = {
    Name        = "corporate-vpc"
    Environment = "production"
    Terraform   = "true"
    Project     = "corporate"
  }
}