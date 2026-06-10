#write a boto3 code for s3 bucket creation and deletion as per your choice with bucket name
import boto3

s3 = boto3.client('s3')
def create_bucket(bucket_name):
    try:
        s3.create_bucket(Bucket=bucket_name)
        print(f'Bucket {bucket_name} created successfully')
    except Exception as e:
        print(f'Error creating bucket {bucket_name}: {str(e)}')

def delete_bucket(bucket_name):
    try:
        s3.delete_bucket(Bucket=bucket_name)
        print(f'Bucket {bucket_name} deleted successfully')
    except Exception as e:
        print(f'Error deleting bucket {bucket_name}: {str(e)}')

# Usage

choice = input("enter yout choise for bucket creation.\n1. creation. \n2. deletion.\n")
bucket = input("enter your bucket name.")
if choice == '1':
    create_bucket(bucket)
elif choice == '2':
    delete_bucket(bucket)