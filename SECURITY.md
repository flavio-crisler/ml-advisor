# Security Policy

ML Advisor is a local, single-user pre-release application. It has no authentication and is not designed to be reachable from an untrusted network. Keep published ports bound to `127.0.0.1`, use test data for public demonstrations, and keep provider keys in the ignored `.env` file.

## Reporting a vulnerability

Please use GitHub's **Report a vulnerability** (private vulnerability reporting) for this repository when available. If it is not enabled, contact the maintainer privately before opening a public issue. Do not post credentials, private datasets, exploit details, or sensitive logs in public issues.

Include the affected component, reproduction steps using synthetic data, impact, and a suggested mitigation if known. The maintainer will acknowledge and triage reports as capacity allows; this project does not currently promise a fixed response time or security-support window.

## Supported versions

The first stable release has not been tagged. Only the current `main` branch is under active development; older commits and snapshots are not maintained security releases.
