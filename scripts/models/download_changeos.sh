#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output_dir="${repo_root}/backend/outputs/changeos"
output_file="${output_dir}/changeos_r34.pt"
download_file="${output_file}.download"
url="https://github.com/Z-Zheng/ChangeOS/releases/download/v0.2/changeos_r34.pt"
sha256="3aa0520284ba5b2358b695732902295ef5fbc225b75ff1885a9eb41609c038d2"

mkdir -p "${output_dir}"
curl -L --fail --retry 2 -o "${download_file}" "${url}"
echo "${sha256}  ${download_file}" | sha256sum --check
mv "${download_file}" "${output_file}"
echo "ChangeOS-R34 weights installed at ${output_file}"
