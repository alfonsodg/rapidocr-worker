#!/usr/bin/env bash
# Rejects staged files inside hidden directories (only .git/.github
# allowed) and staged hidden files outside the allowlist below.
set +e

allowed_dirs=" .git .github "
errors=0

# Allowed hidden basenames; any other dotfile is rejected.
is_allowed_file() {
  case "$1" in
    .gitignore|.gitignore.template|.gitattributes|\
    .editorconfig|.editorconfig.template|.pre-commit-config.yaml|\
    .dockerignore|.env.example|.*.example|.gitkeep|.markdownlint.json|\
    .python-version|.nvmrc|.npmrc|.triage-policies.yml) return 0 ;;
    *) return 1 ;;
  esac
}

# Return 0 if the segment is allowed as a hidden directory.
is_allowed_dir() {
  case "${allowed_dirs}" in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

# Check staged paths for hidden directory segments.
check_staged_paths() {
  local path seg i len
  while read -r path; do
    [ -z "${path}" ] && continue
    IFS='/' read -ra parts <<< "${path}"
    len=${#parts[@]}
    for ((i = 0; i < len - 1; i++)); do
      seg="${parts[$i]}"
      if ! is_allowed_dir "${seg}" && [ "${seg#.}" != "${seg}" ]; then
        echo "ERROR: hidden directory segment '${seg}' is not allowed in staged path: ${path}" >&2
        errors=$((errors + 1))
        break
      fi
    done
    base="${parts[$((len - 1))]}"
    if [ "${base#.}" != "${base}" ] && ! is_allowed_file "${base}"; then
      echo "ERROR: hidden file '${base}' is not in the allowlist: ${path}" >&2
      errors=$((errors + 1))
    fi
  done < <(git diff --cached --name-only --diff-filter=AMR)
}

# Check working tree for hidden dirs that are not ignored by git.
check_working_tree() {
  local dir seg rel
  while IFS= read -r dir; do
    [ -z "${dir}" ] && continue
    rel="${dir#./}"
    seg="${rel%%/*}"
    [ "${seg}" = "" ] || [ "${seg}" = "." ] && continue
    if is_allowed_dir "${seg}" || [ "${seg#.}" = "${seg}" ]; then
      continue
    fi
    if git check-ignore -q "${dir}" 2>/dev/null; then
      continue
    fi
    echo "ERROR: hidden directory is not ignored and must not be committed: ${dir}" >&2
    errors=$((errors + 1))
  done < <(find . -maxdepth 2 -type d -name '.*' 2>/dev/null | sort -u)
}

check_staged_paths
check_working_tree

if [ "${errors}" -gt 0 ]; then
  echo "ERROR: ${errors} hidden directory violation(s) found. Allowed: .git, .github" >&2
  exit 1
fi

exit 0
