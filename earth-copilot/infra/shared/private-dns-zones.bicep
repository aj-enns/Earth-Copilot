@description('Tags to apply to all resources')
param tags object

@description('VNet ID to link DNS zones to')
param vnetId string

@description('Cloud environment: Commercial or Government')
@allowed(['Commercial', 'Government'])
param cloudEnvironment string = 'Commercial'

// DNS zone suffixes differ between Commercial and Government clouds
var isCommercial = cloudEnvironment == 'Commercial'

var dnsZoneNames = {
  containerRegistry: 'privatelink.azurecr.io'
  keyVault: isCommercial ? 'privatelink.vaultcore.azure.net' : 'privatelink.vaultcore.usgovcloudapi.net'
  storageBlob: isCommercial ? 'privatelink.blob.core.windows.net' : 'privatelink.blob.core.usgovcloudapi.net'
  storageFile: isCommercial ? 'privatelink.file.core.windows.net' : 'privatelink.file.core.usgovcloudapi.net'
  cognitiveServices: isCommercial ? 'privatelink.cognitiveservices.azure.com' : 'privatelink.cognitiveservices.azure.us'
  openai: isCommercial ? 'privatelink.openai.azure.com' : 'privatelink.openai.azure.us'
  servicesAi: 'privatelink.services.ai.azure.com'
  mlWorkspace: isCommercial ? 'privatelink.api.azureml.ms' : 'privatelink.api.azureml.us'
  mlNotebooks: isCommercial ? 'privatelink.notebooks.azure.net' : 'privatelink.notebooks.azure.us'
}

resource containerRegistryDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: dnsZoneNames.containerRegistry
  location: 'global'
  tags: tags
}

resource keyVaultDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: dnsZoneNames.keyVault
  location: 'global'
  tags: tags
}

resource storageBlobDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: dnsZoneNames.storageBlob
  location: 'global'
  tags: tags
}

resource storageFileDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: dnsZoneNames.storageFile
  location: 'global'
  tags: tags
}

resource cognitiveServicesDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: dnsZoneNames.cognitiveServices
  location: 'global'
  tags: tags
}

resource openaiDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: dnsZoneNames.openai
  location: 'global'
  tags: tags
}

resource servicesAiDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: dnsZoneNames.servicesAi
  location: 'global'
  tags: tags
}

resource mlWorkspaceDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: dnsZoneNames.mlWorkspace
  location: 'global'
  tags: tags
}

resource mlNotebooksDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: dnsZoneNames.mlNotebooks
  location: 'global'
  tags: tags
}

// Link all DNS zones to the VNet
resource containerRegistryLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: containerRegistryDnsZone
  name: 'link-cr'
  location: 'global'
  properties: {
    virtualNetwork: { id: vnetId }
    registrationEnabled: false
  }
}

resource keyVaultLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: keyVaultDnsZone
  name: 'link-kv'
  location: 'global'
  properties: {
    virtualNetwork: { id: vnetId }
    registrationEnabled: false
  }
}

resource storageBlobLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: storageBlobDnsZone
  name: 'link-st-blob'
  location: 'global'
  properties: {
    virtualNetwork: { id: vnetId }
    registrationEnabled: false
  }
}

resource storageFileLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: storageFileDnsZone
  name: 'link-st-file'
  location: 'global'
  properties: {
    virtualNetwork: { id: vnetId }
    registrationEnabled: false
  }
}

resource cognitiveServicesLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: cognitiveServicesDnsZone
  name: 'link-cog'
  location: 'global'
  properties: {
    virtualNetwork: { id: vnetId }
    registrationEnabled: false
  }
}

resource openaiLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: openaiDnsZone
  name: 'link-oai'
  location: 'global'
  properties: {
    virtualNetwork: { id: vnetId }
    registrationEnabled: false
  }
}

resource servicesAiLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: servicesAiDnsZone
  name: 'link-sai'
  location: 'global'
  properties: {
    virtualNetwork: { id: vnetId }
    registrationEnabled: false
  }
}

resource mlWorkspaceLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: mlWorkspaceDnsZone
  name: 'link-ml'
  location: 'global'
  properties: {
    virtualNetwork: { id: vnetId }
    registrationEnabled: false
  }
}

resource mlNotebooksLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: mlNotebooksDnsZone
  name: 'link-nb'
  location: 'global'
  properties: {
    virtualNetwork: { id: vnetId }
    registrationEnabled: false
  }
}

output containerRegistryDnsZoneId string = containerRegistryDnsZone.id
output keyVaultDnsZoneId string = keyVaultDnsZone.id
output storageBlobDnsZoneId string = storageBlobDnsZone.id
output storageFileDnsZoneId string = storageFileDnsZone.id
output cognitiveServicesDnsZoneId string = cognitiveServicesDnsZone.id
output openaiDnsZoneId string = openaiDnsZone.id
output servicesAiDnsZoneId string = servicesAiDnsZone.id
output mlWorkspaceDnsZoneId string = mlWorkspaceDnsZone.id
output mlNotebooksDnsZoneId string = mlNotebooksDnsZone.id
