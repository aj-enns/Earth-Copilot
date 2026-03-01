param location string = resourceGroup().location
param tags object = {}

param staticWebAppName string
param containerAppUrl string = ''

@description('SKU for the Static Web App (Free or Standard)')
param sku object = {
  name: 'Free'
  tier: 'Free'
}

// Azure Static Web App — hosts the React SPA with zero App Service VM quota required
resource staticWebApp 'Microsoft.Web/staticSites@2023-01-01' = {
  name: staticWebAppName
  location: location
  tags: tags
  sku: sku
  properties: {
    stagingEnvironmentPolicy: 'Enabled'
    allowConfigFileUpdates: true
    buildProperties: {
      skipGithubActionWorkflowGeneration: true
    }
  }
}

// App settings for the Static Web App
resource staticWebAppSettings 'Microsoft.Web/staticSites/config@2023-01-01' = {
  parent: staticWebApp
  name: 'appsettings'
  properties: {
    VITE_API_BASE_URL: containerAppUrl
  }
}

// Note: Microsoft Entra ID authentication can be configured via staticwebapp.config.json
// or manually in the Azure Portal under the Static Web App's Authentication settings.

output staticWebAppName string = staticWebApp.name
output staticWebAppId string = staticWebApp.id
output staticWebAppUrl string = 'https://${staticWebApp.properties.defaultHostname}'
output staticWebAppDeploymentToken string = staticWebApp.listSecrets().properties.apiKey
