@description('Name for the private endpoint')
param name string

@description('Azure region')
param location string

@description('Tags to apply')
param tags object

@description('Resource ID of the service to connect to')
param serviceResourceId string

@description('Sub-resource group ID (e.g. registry, vault, blob, account, amlworkspace)')
param groupId string

@description('Subnet ID to place the private endpoint in')
param subnetId string

@description('Primary private DNS zone ID for A-record registration')
param privateDnsZoneId string

@description('Additional private DNS zone IDs (e.g. OpenAI, Notebooks)')
param additionalDnsZoneIds array = []

// Build the full list of DNS zone configs
var primaryDnsConfig = [
  {
    id: privateDnsZoneId
  }
]

var additionalDnsConfigs = [for zoneId in additionalDnsZoneIds: {
  id: zoneId
}]

resource privateEndpoint 'Microsoft.Network/privateEndpoints@2023-11-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    subnet: {
      id: subnetId
    }
    privateLinkServiceConnections: [
      {
        name: '${name}-connection'
        properties: {
          privateLinkServiceId: serviceResourceId
          groupIds: [
            groupId
          ]
        }
      }
    ]
  }
}

resource dnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-11-01' = {
  parent: privateEndpoint
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [for (zone, i) in concat(primaryDnsConfig, additionalDnsConfigs): {
      name: 'config-${i}'
      properties: {
        privateDnsZoneId: zone.id
      }
    }]
  }
}
