# Use an official Python runtime as a parent image
FROM public.ecr.aws/lambda/python:3.10

# Copy the entire project into the container
COPY . ${LAMBDA_TASK_ROOT}

# Copy your requirements.txt into the container and install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

WORKDIR ${LAMBDA_TASK_ROOT}

# Set the entry point and handler for the Lambda function
CMD ["generate.lambda_handler"]
