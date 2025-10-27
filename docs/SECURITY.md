# Security & Secrets

**Never commit real API tokens or credentials to Git.** This repository loads secrets from a local `.env` file (ignored by `.gitignore`).

## ENTSO-E token
- Keep it in `.env` as `ENTSOE_TOKEN=...`
- Do *not* paste it into `.env.example` or README.
- If a token is ever exposed publicly, **revoke and rotate immediately** in the ENTSO-E account settings, then re-run data fetch.

## Local verification
Use:
```bash
python scripts/check_token.py
```
It calls a tiny date window on the ENTSO-E API and prints `OK` or a helpful error.

## Cleaning a leaked secret from history
If a secret was committed, you must remove it from history (not just a new commit):
- Easiest: GitHub → **Settings → Secret scanning**/**Security** tools, or
- Use **BFG Repo-Cleaner** or **git filter-repo** to purge the file/line, force-push, and invalidate caches.

````
# Example with git filter-repo (install first)
# Remove any .env.example containing a token across history
python -m pip install git-filter-repo
git clone --mirror https://github.com/<you>/irish-dam-forecast.git
cd irish-dam-forecast.git
# Remove the specific line pattern (simple approach):
git filter-repo --path .env.example --invert-paths
# or rewrite file content with a clean template if desired
# Then force-push
git push --force --tags origin --all
````
