# On CI
## build image and push (local machine)
sudo make run-build
sudo docker tag llmstudio-llmstudio llmstudio.azurecr.io/llmstudio:latest
az acr login --name llmstudio
sudo docker push llmstudio.azurecr.io/llmstudio:latest

## check state
az container show --resource-group llmstudio --name llmstudio --query instanceView.state

## On CD
ssh to the machine:
`ssh -i ~/.ssh/llmstudio-vm.pem llmstudio@4.180.10.0`

`cd llmstudio_deploy`

`sudo make run-build`
