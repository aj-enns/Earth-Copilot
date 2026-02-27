@description('Name for the Bot Service')
param name string

@description('Tags to apply')
param tags object

@description('Microsoft App ID (from Entra App Registration)')
param microsoftAppId string

@description('Messaging endpoint URL')
param messagingEndpoint string

@description('Tenant ID')
param tenantId string

resource botService 'Microsoft.BotService/botServices@2022-09-15' = {
  name: name
  location: 'global'
  tags: tags
  sku: {
    name: 'S1'
  }
  kind: 'azurebot'
  properties: {
    displayName: name
    endpoint: messagingEndpoint
    msaAppId: microsoftAppId
    msaAppTenantId: tenantId
    msaAppType: 'SingleTenant'
    schemaTransformationVersion: '1.3'
  }
}

// Enable Teams channel
resource teamsChannel 'Microsoft.BotService/botServices/channels@2022-09-15' = {
  parent: botService
  name: 'MsTeamsChannel'
  location: 'global'
  properties: {
    channelName: 'MsTeamsChannel'
    properties: {
      isEnabled: true
    }
  }
}

output botServiceName string = botService.name
