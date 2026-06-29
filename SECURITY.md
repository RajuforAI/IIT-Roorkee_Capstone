# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 1.0.x | ✅ |
| < 1.0 | ❌ |

Only the latest minor line receives security backports. Older lines are
end-of-life and should be upgraded.

## Reporting a Vulnerability

Please report security issues **privately** to:

**security@telegenie-ai.example.com**[^1]

A GPG-encrypted reply is available on request — include that ask in your first
message and we will send a public key fingerprint back.

You can expect an initial acknowledgement within **5 business days**. We will
follow up with a triage assessment and a coordinated disclosure timeline
within 10 business days of the initial report.

## Disclosure Policy

We follow a **coordinated disclosure** model with a **90-day disclosure
window**. After a fix is released we publish a post-mortem describing the
issue, the affected versions, the fix version, and credit to the reporter
(unless they ask to remain anonymous).

If a report requires longer than 90 days to remediate, we will negotiate an
extension with the reporter before the window closes.

## What to Include

To help us triage quickly, please include:

- **Affected version** (commit SHA or release tag)
- **Attack vector** — what input, request, or environment state triggers it
- **Impact** — what an attacker can achieve (RCE, data exfiltration, auth bypass, etc.)
- **Reproduction steps** — minimal steps to demonstrate the issue
- **Your contact** — email or other channel for follow-up questions

## Hall of Fame

We thank the following researchers for privately disclosing vulnerabilities to
the TeleGenie AI project. *(No reports yet — be the first!)*

- _Your name here_

## Out of Scope

The following are tracked elsewhere and **not** accepted as security reports
against this repository:

- **Denial of service via OpenAI / Gemini quota exhaustion** — this is an
  operational / billing concern owned by the deployer. Configure the daily USD
  cap in the admin panel and subscribe to your provider's billing alerts.
- **Dependency CVEs** — third-party library vulnerabilities are tracked via
  Dependabot and patched in regular dependency-update PRs. Open a regular
  issue if Dependabot misses one.
- **Social engineering** of project maintainers
- **Volumetric / network-layer DDoS** against the deployed instance — handle
  upstream (CloudFront / ALB / WAF)

[^1]: This is a placeholder security contact until a monitored inbox is
provisioned. If the placeholder bounces, open a private GitHub Security
Advisory on the repository instead.