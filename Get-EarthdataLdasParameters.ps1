param(
    [string]$OutputCsv = "",
    [switch]$IncludeCoordinateVariables,
    [ValidateSet('All', 'NLDAS', 'GLDAS')]
    [string]$Dataset = 'All',
    [int]$BatchSize = 50
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$collectionsToQuery = @(
    @{
        Label = 'NLDAS'
        Provider = 'GES_DISC'
        ShortName = 'NLDAS_FORA0125_H'
        Version = '2.0'
    },
    @{
        Label = 'GLDAS'
        Provider = 'GES_DISC'
        ShortName = 'GLDAS_NOAH025_3H'
        Version = '2.1'
    }
)

if ($Dataset -ne 'All') {
    $collectionsToQuery = @($collectionsToQuery | Where-Object { $_.Label -eq $Dataset })
}

function Get-CmrCollection {
    param(
        [Parameter(Mandatory)] [string]$Provider,
        [Parameter(Mandatory)] [string]$ShortName,
        [Parameter(Mandatory)] [string]$Version
    )

    $url = "https://cmr.earthdata.nasa.gov/search/collections.json?provider=$([uri]::EscapeDataString($Provider))&short_name=$([uri]::EscapeDataString($ShortName))&version=$([uri]::EscapeDataString($Version))&page_size=10"
    $response = Invoke-RestMethod -Uri $url -Method Get

    if (-not $response.feed.entry -or $response.feed.entry.Count -eq 0) {
        throw "No collection found for $ShortName version $Version from provider $Provider."
    }

    return $response.feed.entry[0]
}

function Get-CmrVariablesByConceptIds {
    param(
        [Parameter(Mandatory)] [string[]]$ConceptIds,
        [int]$BatchSize = 50
    )

    $allItems = @()
    for ($start = 0; $start -lt $ConceptIds.Count; $start += $BatchSize) {
        $end = [Math]::Min($start + $BatchSize - 1, $ConceptIds.Count - 1)
        $batch = $ConceptIds[$start..$end]
        $queryString = ($batch | ForEach-Object {
            'concept_id[]=' + [uri]::EscapeDataString($_)
        }) -join '&'

        $url = "https://cmr.earthdata.nasa.gov/search/variables.json?$queryString&page_size=$BatchSize"
        $response = Invoke-RestMethod -Uri $url -Method Get
        if ($response.items) {
            $allItems += $response.items
        }
    }

    return $allItems
}

$results = foreach ($target in $collectionsToQuery) {
    Write-Host "Querying $($target.Label) collection metadata..." -ForegroundColor Cyan
    $collection = Get-CmrCollection -Provider $target.Provider -ShortName $target.ShortName -Version $target.Version

    $variableIds = @($collection.association_details.variables | ForEach-Object { $_.concept_id })
    if (-not $variableIds -or $variableIds.Count -eq 0) {
        Write-Warning "No variable associations found for $($target.ShortName)."
        continue
    }

    $variables = Get-CmrVariablesByConceptIds -ConceptIds $variableIds -BatchSize $BatchSize

    foreach ($variable in $variables) {
        $isCoordinate = $variable.name -in @('time', 'lat', 'lon')
        if (-not $IncludeCoordinateVariables -and $isCoordinate) {
            continue
        }

        [pscustomobject]@{
            DatasetLabel         = $target.Label
            CollectionShortName  = $collection.short_name
            CollectionVersion    = $collection.version_id
            CollectionConceptId  = $collection.id
            CollectionEntryId    = $collection.entry_id
            VariableConceptId    = $variable.concept_id
            VariableName         = $variable.name
            LongName             = $variable.long_name
            Definition           = $variable.definition
            NativeId             = $variable.native_id
            Provider             = $variable.provider_id
        }
    }
}

$results = $results | Sort-Object DatasetLabel, VariableName

Write-Host ''
Write-Host "$Dataset Earthdata parameters:" -ForegroundColor Green
$results | Format-Table DatasetLabel, VariableName, LongName, VariableConceptId -AutoSize

if ($OutputCsv) {
    $results | Export-Csv -Path $OutputCsv -NoTypeInformation -Encoding UTF8
    Write-Host "`nSaved CSV to: $OutputCsv" -ForegroundColor Yellow
}

$results


