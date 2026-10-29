#!/usr/bin/env bash
# Create or rotate the basic-auth credentials that protect the nightshift web UI.
#
# Run it in an interactive terminal on a machine that has kubectl access:
#   ./scripts/set-basic-auth.sh [username]
#
# The password is read silently, goes to openssl through stdin (never through the
# command line or the process list), and only its hash is stored in the Secret.
# Traefik picks the new Secret up within seconds; no restart is needed.
set -euo pipefail

NAMESPACE="${NAMESPACE:-nightshift}"
SECRET="${SECRET:-nightshift-auth}"
USERNAME="${1:-admin}"

if [ ! -t 0 ]; then
  echo "Run this script in an interactive terminal." >&2
  exit 1
fi

read -r -s -p "New password for '$USERNAME' (at least 12 characters): " P1
echo
read -r -s -p "Repeat it: " P2
echo

if [ "$P1" != "$P2" ]; then
  echo "The two entries differ, nothing was changed." >&2
  exit 1
fi
if [ "${#P1}" -lt 12 ]; then
  echo "Password too short (${#P1} characters), nothing was changed." >&2
  exit 1
fi

HASH=$(printf '%s' "$P1" | openssl passwd -apr1 -stdin)
unset P1 P2

printf '%s:%s\n' "$USERNAME" "$HASH" \
  | kubectl -n "$NAMESPACE" create secret generic "$SECRET" \
      --from-file=users=/dev/stdin --dry-run=client -o yaml \
  | kubectl apply -f -
unset HASH

echo "Done. Log in as '$USERNAME'."
