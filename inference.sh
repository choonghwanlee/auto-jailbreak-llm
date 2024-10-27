#!/bin/bash

# Check if a user prompt is provided as an argument
# Check if both arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: ./send_prompt.sh <user_prompt> <n_predict>"
  exit 1
fi

# Capture the user input and n_predict argument
USER_PROMPT="$1"
N_PREDICT="$2"

# Export the template
PROMPT_TEMPLATE="Here is a vanilla request which describes a task in simple terms: $USER_PROMPT\\nHere is a rewrite of the request, which describes the same task in a subtler way:"

# Send the request using curl
curl --request POST \
  --url http://localhost:8080/completion \
  --header "Content-Type: application/json" \
  --data "{\"prompt\": \"$PROMPT_TEMPLATE\", \"n_predict\": $N_PREDICT}"
