#!/usr/bin/env pwsh

function Install {
    <#
    .SYNOPSIS
        Install the poetry environment and install the pre-commit hooks
    #>
    Write-Host "Creating virtual environment using pyenv and poetry"
    poetry install
    poetry run pre-commit install
    poetry shell
}

function Check {
    <#
    .SYNOPSIS
        Run code quality tools.
    #>
    Write-Host "Checking Poetry lock file consistency with 'pyproject.toml': Running poetry check --lock"
    poetry check --lock
    Write-Host "Linting code: Running pre-commit"
    poetry run pre-commit run -a
    Write-Host "Static type checking: Running mypy"
    poetry run mypy
}

function Test {
    <#
    .SYNOPSIS
        Test the code with pytest
    #>
    Write-Host "Testing code: Running pytest"
    poetry run pytest --cov --cov-config=pyproject.toml --cov-report=xml
}

function CleanBuild {
    <#
    .SYNOPSIS
        Clean build artifacts
    #>
    if (Test-Path -Path "dist") {
        Write-Host "Cleaning build directory..."
        Remove-Item -Path "dist" -Recurse -Force
    }
}

function Build {
    <#
    .SYNOPSIS
        Build wheel file using poetry
    #>
    CleanBuild
    Write-Host "Creating wheel file"
    poetry build
}

function Publish {
    <#
    .SYNOPSIS
        Publish a release to pypi.
    #>
    Write-Host "Publishing: Dry run."

    # Check if PYPI_TOKEN environment variable exists
    if (-not (Test-Path env:PYPI_TOKEN)) {
        Write-Error "PYPI_TOKEN environment variable is not set. Please set it before publishing:  $env:PYPI_TOKEN = "your-token-here""
        return
    }

    poetry config pypi-token.pypi $env:PYPI_TOKEN
    poetry publish --dry-run
    Write-Host "Publishing."
    poetry publish
}

function BuildAndPublish {
    <#
    .SYNOPSIS
        Build and publish.
    #>
    Build
    Publish
}

function ShowHelp {
    <#
    .SYNOPSIS
        Show help information
    #>
    Write-Host "Available commands:"
    Write-Host "  Install           - Install the poetry environment and install the pre-commit hooks"
    Write-Host "  Check             - Run code quality tools"
    Write-Host "  Test              - Test the code with pytest"
    Write-Host "  Build             - Build wheel file using poetry"
    Write-Host "  CleanBuild        - Clean build artifacts"
    Write-Host "  Publish           - Publish a release to pypi"
    Write-Host "  BuildAndPublish   - Build and publish"
    Write-Host "  ShowHelp          - Show this help information"
}

# Process command line arguments
if ($args.Count -eq 0) {
    ShowHelp
    return
}

switch ($args[0].ToLower()) {
    "install" { Install }
    "check" { Check }
    "test" { Test }
    "build" { Build }
    "clean-build" { CleanBuild }
    "publish" { Publish }
    "build-and-publish" { BuildAndPublish }
    "help" { ShowHelp }
    default {
        Write-Host "Unknown command: $($args[0])"
        ShowHelp
    }
}
