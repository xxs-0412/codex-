# GitHub Backup Workflow

This repo is configured to avoid overwriting the previous GitHub snapshot.

## What changed

- Pushes to `main` and `master` are blocked by a local `pre-push` hook.
- Use `工具和杂项/脚本与配置/上传GitHub备份.bat` to upload a backup.
- Each backup goes to a new branch named like `backup/20260413_101530_ab12cd3`.

## How to use it

1. Commit your changes locally.
2. Run `工具和杂项/脚本与配置/上传GitHub备份.bat`.
3. The script creates and pushes a new timestamped backup branch.

## Notes

- The script stops if there are uncommitted changes.
- Existing GitHub backups stay in place because each upload uses a new branch.
