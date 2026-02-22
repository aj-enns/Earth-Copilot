param location string = resourceGroup().location
param tags object = {}
param name string

@description('Enable private endpoints — requires Premium SKU')
param enablePrivateEndpoints bool = false

@description('Subnet ID for the ACR agent pool (VNet-integrated builds)')
param agentPoolSubnetId string = ''

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: enablePrivateEndpoints ? 'Premium' : 'Basic'  // Premium required for private endpoints
  }
  properties: {
    adminUserEnabled: false  // Use managed identity instead of admin user
    anonymousPullEnabled: false  // Security best practice
    publicNetworkAccess: enablePrivateEndpoints ? 'Disabled' : 'Enabled'
    networkRuleBypassOptions: enablePrivateEndpoints ? 'AzureServices' : 'AzureServices'
  }
}

// VNet-integrated agent pool for building images without public ACR access
resource agentPool 'Microsoft.ContainerRegistry/registries/agentPools@2019-06-01-preview' = if (enablePrivateEndpoints && !empty(agentPoolSubnetId)) {
  parent: registry
  name: 'buildpool'
  location: location
  properties: {
    count: 1
    tier: 'S1'
    os: 'Linux'
    virtualNetworkSubnetResourceId: agentPoolSubnetId
  }
}

output name string = registry.name
output loginServer string = registry.properties.loginServer
output id string = registry.id
