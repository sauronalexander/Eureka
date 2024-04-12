import os
import boto3


def set_message_to_training_queue(message):
    sqs = boto3.client('sqs', region_name=os.getenv("AWS_REGION"))  # Replace 'your-region' with your AWS region

    # Specify the URL of the SQS queue
    queue_url = os.getenv("EUREKA_TRAINING_QUEUE_URL")

    # Specify the message to be sent

    # Send the message to the SQS queue
    response = sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=message
    )

    # Print the message ID and MD5 digest (optional)
    print(f"MessageId: {response['MessageId']}")
    print(f"MD5OfMessageBody: {response['MD5OfMessageBody']}")
