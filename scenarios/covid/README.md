# COVID predictive modelling 

This hypothetical scenario involves three training data providers (TDPs), ICMR, COWIN and a state war room, and a TDC who wishes the train a model using datasets from these TDPs. The repository contains sample datasets and a model. The model and datasets are for illustrative purposes only; none of these organizations have been involved in contributing to the code or datasets.  

The end-to-end training pipeline consists of the following phases. 

1. Data pre-processing and de-identification
2. Data packaging, encryption and upload
3. Model packaging, encryption and upload 
4. Encryption key import with key release policies
5. Deployment and execution of CCR
6. Model decryption 

## Pre-requisites

### GitHub Codespaces

The simplest way to setup a development environment is using [GitHub Codespaces](https://github.com/codespaces). The repository includes a [devcontainer.json](../../.devcontainer/devcontainer.json), which customizes your codespace to install all required dependencies. 

### Local Development Environment

Alternatively, you deploy the sample locally on Linux (we have tested with Ubuntu 20.04), or Windows with WSL 2. You will need to install the following dependencies. 

- [docker](https://docs.docker.com/engine/install/ubuntu/) and docker-compose. After installing docker, add your user to the docker group using `sudo usermod -aG docker $USER`, and log back in to a shell. 
- make (install using ```sudo apt-get install make```)
- Python 3.6.9 and pip 
- Python wheel package (install using ```pip install wheel```)

## Build CCR containers

To build your own CCR container images, use the following command from the root of the repository. 

```bash
./ci/build.sh
cd scenarios/covid
./ci/build.sh
```

These scripts build the following containers. 

- ```depa-training```: Container with the core CCR logic for joining datasets and running differentially private training. 
- ```depa-training-encfs```: Container for loading encrypted data into the CCR. 
- ```preprocess-icmr, preprocess-cowin, preprocess-index```: Containers that pre-process and de-identify datasets. 
- ```ccr-model-save```: Container that saves the model to be trained in ONNX format. 

Alternatively, you can use pre-built container images from the ```ispirt``` repository by setting the following environment variable. 
```bash
export CONTAINER_REGISTRY=ispirt
```

## Data pre-processing and de-identification

The folders ```scenarios/covid/data``` contains three sample training datasets. Acting as TDPs for these datasets, run the following scripts to de-identify the datasets. 

```bash
cd scenarios/covid/deployment/docker
./preprocess.sh
```

This script performs pre-processing and de-identification of these datasets before sharing with the TDC.

## Prepare model for training

Next, acting as a TDC, save a sample model using the following script. 

```bash
./save-model.sh
```

This script will save the model as ```scenarios/covid/data/modeller/model/model.onnx.```

## Deploy locally

Assuming you have cleartext access to all the de-identified datasets, you can train the model as follows. 

```bash
./train.sh
```
The script joins the datasets using a configuration defined in [query_config.json](./config/query_config.json) and trains the model using a configuration defined in [model_config.json](./config/model_config.json). If all goes well, you should see output similar to the following output, and the trained model will be saved under the folder `/tmp/output`. 

```
docker-train-1  | {'input_dataset_path': '/tmp/sandbox_icmr_cowin_index_without_key_identifiers.csv', 'saved_model_path': '/mnt/remote/model/model.onnx', 'saved_model_optimizer': '/mnt/remote/model/dpsgd_model_opimizer.pth', 'saved_weights_path': '', 'batch_size': 2, 'total_epochs': 5, 'max_grad_norm': 0.1, 'epsilon_threshold': 1.0, 'delta': 0.01, 'sample_size': 60000, 'target_variable': 'icmr_a_icmr_test_result', 'test_train_split': 0.2, 'metrics': ['accuracy', 'precision', 'recall']}
docker-train-1  | Epoch [1/5], Loss: 0.0084
docker-train-1  | Epoch [2/5], Loss: 0.4231
docker-train-1  | Epoch [3/5], Loss: 0.0008
docker-train-1  | Epoch [4/5], Loss: 0.0138
docker-train-1  | Epoch [5/5], Loss: 0.0489
```

## Deploy to Azure

In a more realistic scenario, these datasets will not be available in the clear to the TDC, and the TDC will be required to use a CCR for training her model. The following steps describe the process of sharing encrypted datasets with TDCs and setting up a CCR in Azure for training models. Please stay tuned for CCR on other cloud platforms. 

To deploy in Azure, you will need the following. 

- Docker Hub account to store container images. Alternatively, you can use pre-built images from the ```ispirt``` container registry. 
- [Azure Key Vault](https://azure.microsoft.com/en-us/products/key-vault/) to store encryption keys and implement secure key release to CCR. You can either you Azure Key Vault Premium (lower cost), or [Azure Key Vault managed HSM](https://learn.microsoft.com/en-us/azure/key-vault/managed-hsm/overview) for enhanced security. Please see instructions below on how to create and setup your AKV instance. 
- Valid Azure subscription with sufficient access to create key vault, storage accounts, storage containers, and Azure Container Instances. 

If you are using your own development environment instead of a dev container or codespaces, you will to install the following dependencies. 

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux).  
- [Azure CLI Confidential containers extension](https://learn.microsoft.com/en-us/cli/azure/confcom?view=azure-cli-latest). After installing Azure CLI, you can install this extension using ```az extension add --name confcom -y```
- [Go](https://go.dev/doc/install). Follow the instructions to install Go. After installing, ensure that the PATH environment variable is set to include ```go``` runtime.
- ```jq```. You can install jq using ```sudo apt-get install -y jq```

We will be creating the following resources as part of the deployment. 

- Azure Key Vault
- Azure Storage account
- Storage containers to host encrypted datasets
- Azure Container Instances to deploy the CCR and train the model

### Push Container Images

If you wish to use your own container images, login to docker hub and push containers to your container registry. 

> **Note:** Replace `<docker-hub-registry-name>` the name of your docker hub registry name.

```bash
export CONTAINER_REGISTRY=<docker-hub-registry-name>
docker login 
./ci/push-containers.sh
cd scenarios/covid
./ci/push-containers.sh
```

### Create Resources

Acting as the TDP, we will create a resource group, a key vault instance and storage containers to host encrypted training datasets and encryption keys. In a real deployments, TDPs and TDCs will use their own key vault instance. However, for this sample, we will use one key vault instance to store keys for all datasets and models. 

> **Note:** At this point, automated creation of AKV managed HSMs is not supported. 

> **Note:** Replace `<resource-group-name>` and `<key-vault-endpoint>` with names of your choice. Storage account names must not container any special characters. Key vault endpoints are of the form `<key-vault-name>.vault.azure.net` (for Azure Key Vault Premium) and `<key-vault-name>.managedhsm.azure.net` for AKV managed HSM, **with no leading https**. This endpoint must be the same endpoint you used while creating the contract.

```bash
az login

export AZURE_RESOURCE_GROUP=<resource-group-name>
export AZURE_KEYVAULT_ENDPOINT=<key-vault-endpoint>
export AZURE_STORAGE_ACCOUNT_NAME=<unique-storage-account-name>
export AZURE_ICMR_CONTAINER_NAME=icmrcontainer
export AZURE_COWIN_CONTAINER_NAME=cowincontainer
export AZURE_INDEX_CONTAINER_NAME=indexcontainer
export AZURE_MODEL_CONTAINER_NAME=modelcontainer
export AZURE_OUTPUT_CONTAINER_NAME=outputcontainer

cd scenarios/covid/data
./1-create-storage-containers.sh 
./2-create-akv.sh
```

### Sign and Register Contract

Next, follow instructions [here](./../../external/contract-ledger/README.md) to sign and register a contract the contract service. The registered contract must contain references to the datasets with matching names, keyIDs and Azure Key Vault endpoints used in this sample. A sample contract is provided [here](https://github.com/kapilvgit/contract-ledger/blob/main/demo/contract/contract.json). After signing and registering the contract, retain the contract service URL and sequence number of the contract for the rest of this sample. 

### Import encryption keys

Next, use the following script to generate and import encryption keys into Azure Key Vault with a policy based on [policy-in-template.json](./policy/policy-in-template.json). The policy requires that the CCRs run specific containers with a specific configuration which includes the public identity of the contract service. Only CCRs that satisfy this policy will be granted access to the encryption keys. 

> **Note:** Replace `<repo-root>` with the path to and including the `depa-training` folder where the repository was cloned. 

```bash
export CONTRACT_SERVICE_URL=<contract-service-url>
export TOOLS_HOME=<repo-root>/external/confidential-sidecar-containers/tools
./3-import-keys.sh
```

The generated keys are available as files with the extension `.bin`. 

### Encrypt Datasets and Model

Next, encrypt the datasets and models using keys generated in the previous step. 

```bash
cd scenarios/covid/data
./4-encrypt-data.sh
```

This step will generate five encrypted file system images (with extension `.img`), three for the datasets, one encrypted file system image containing the model, and one image where the trained model will be stored.

### Upload Datasets

Now upload encrypted datasets to Azure storage containers.

```bash
./5-upload-encrypted-data.sh
```

### Deploy CCR in Azure

Acting as a TDC, use the following script to deploy the CCR using Confidential Containers on Azure Container Instances. 

> **Note:** Replace `<contract-sequence-number>` with the sequence number of the contract registered with the contract service. 

```bash
cd scenarios/covid/deployment/aci
./deploy.sh -c <contract-sequence-number> -m ../../config/model_config.json -q ../../config/query_config.json
```

This script will deploy the container images from your container registry, including the encrypted filesystem sidecar. The sidecar will generate an SEV-SNP attestation report, generate an attestation token using the Microsoft Azure Attestation (MAA) service, retrieve dataset, model and output encryption keys from the TDP and TDC's Azure Key Vault, train the model, and save the resulting model into TDC's output filesystem image, which the TDC can later decrypt. 

Once the deployment is complete, you can obtain logs from the CCR using the following commands. Note there may be some delay in getting the logs are deployment is complete. 

```bash
# Obtain logs from the training container
az container logs --name depa-training-covid --resource-group $AZURE_RESOURCE_GROUP --container-name depa-training

# Obtain logs from the encrypted filesystem sidecar
az container logs --name depa-training-covid --resource-group $AZURE_RESOURCE_GROUP --container-name encrypted-storage-sidecar
```

### Download and decrypt trained model

You can download and decrypt the trained model using the following script. 

```bash
cd scenarios/covid/data
./6-download-decrypt-model.sh
```

The trained model is available in `output` folder. 

### Clean-up

You can use the following command to delete the resource group and clean-up all resources used in the demo. 

```bash
az group delete --yes --name $AZURE_RESOURCE_GROUP
```
