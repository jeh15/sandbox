#!/bin/sh

  set -e

  NC='\033[0m' # No Color
  BRIGHT_BLUE='\033[0;94m'
  YELLOW='\033[0;33m'
  RED='\033[0;31m'
  
  # Some utilities from https://github.com/client9/shlib

  echoerr() {
    printf "$@
" 1>&2
  }
  
  log_info() {
    printf "${BRIGHT_BLUE}  ==>${NC} $@
"
  }

  log_warn() {
    printf "${YELLOW}  ==>${NC} $@
"
  }
  
  log_crit() {
    echoerr
    echoerr "  ${RED}$@${NC}"
    echoerr
  }
  
  is_command() {
    command -v "$1" >/dev/null
    #type "$1" > /dev/null 2> /dev/null
  }
  
  http_download_curl() {
    local_file=$1
    source_url=$2
    header1=$3
    header2=$4
    header3=$5

    code=$(curl  -w '%{http_code}' -H "$header1" -H "$header2" -H "$header3" -fsSL -o "$local_file" "$source_url")
    if [ "$code" != "200" ]; then
      log_crit "Error downloading, got $code response from server"
      return 1
    fi
    return 0
  }

  uname_os() {
    os=$(uname -s | tr '[:upper:]' '[:lower:]')
  
    # fixed up for https://github.com/client9/shlib/issues/3
    case "$os" in
      msys_nt*) os="windows" ;;
      mingw*) os="windows" ;;
    esac
  
    # other fixups here
    echo "$os"
  }
  
  uname_os_check() {
    os=$(uname_os)
    case "$os" in
      darwin) return 0 ;;
      dragonfly) return 0 ;;
      freebsd) return 0 ;;
      linux) return 0 ;;
      android) return 0 ;;
      nacl) return 0 ;;
      netbsd) return 0 ;;
      openbsd) return 0 ;;
      plan9) return 0 ;;
      solaris) return 0 ;;
      windows) return 0 ;;
    esac
    log_crit "uname_os_check '$(uname -s)' got converted to '$os' which is not supported by Bina."
    return 1
  }
  
  uname_arch() {
    arch=$(uname -m)
    case $arch in
      x86_64) arch="amd64" ;;
      x86) arch="386" ;;
      i686) arch="386" ;;
      i386) arch="386" ;;
      aarch64) arch="arm64" ;;
      armv5*) arch="armv5" ;;
      armv6*) arch="armv6" ;;
      armv7*) arch="armv7" ;;
    esac
    echo ${arch}
  }
  
  uname_arch_check() {
    arch=$(uname_arch)
      case "$arch" in
      386) return 0 ;;
      amd64) return 0 ;;
      arm64) return 0 ;;
      armv5) return 0 ;;
      armv6) return 0 ;;
      armv7) return 0 ;;
      ppc64) return 0 ;;
      ppc64le) return 0 ;;
      mips) return 0 ;;
      mipsle) return 0 ;;
      mips64) return 0 ;;
      mips64le) return 0 ;;
      s390x) return 0 ;;
      amd64p32) return 0 ;;
    esac
    log_crit "uname_arch_check '$(uname -m)' got converted to '$arch' which is not supported by Bina."
    return 1
  }

  platform_check() {
    platform="$os-$arch"
    case "$platform" in
      unknown-unknown) 
          download_url="https://api.github.com/repos/charmbracelet/gum/releases/assets/91458490" 
          download_file_name="gum-0.9.0.tar.gz"
          bin_file="gum"
          ;;
darwin-arm64) 
          download_url="https://api.github.com/repos/charmbracelet/gum/releases/assets/91458485" 
          download_file_name="gum_0.9.0_Darwin_arm64.tar.gz"
          bin_file="gum"
          ;;
darwin-amd64) 
          download_url="https://api.github.com/repos/charmbracelet/gum/releases/assets/91458484" 
          download_file_name="gum_0.9.0_Darwin_x86_64.tar.gz"
          bin_file="gum"
          ;;
unknown-arm64) 
          download_url="https://api.github.com/repos/charmbracelet/gum/releases/assets/91458476" 
          download_file_name="gum_0.9.0_freebsd_arm64.tar.gz"
          bin_file="gum"
          ;;
unknown-armv7) 
          download_url="https://api.github.com/repos/charmbracelet/gum/releases/assets/91458486" 
          download_file_name="gum_0.9.0_freebsd_armv7.tar.gz"
          bin_file="gum"
          ;;
unknown-386) 
          download_url="https://api.github.com/repos/charmbracelet/gum/releases/assets/91458481" 
          download_file_name="gum_0.9.0_freebsd_i386.tar.gz"
          bin_file="gum"
          ;;
unknown-amd64) 
          download_url="https://api.github.com/repos/charmbracelet/gum/releases/assets/91458475" 
          download_file_name="gum_0.9.0_freebsd_x86_64.tar.gz"
          bin_file="gum"
          ;;
linux-arm64) 
          download_url="https://api.github.com/repos/charmbracelet/gum/releases/assets/91458482" 
          download_file_name="gum_0.9.0_Linux_arm64.tar.gz"
          bin_file="gum"
          ;;
linux-armv7) 
          download_url="https://api.github.com/repos/charmbracelet/gum/releases/assets/91458483" 
          download_file_name="gum_0.9.0_Linux_armv7.tar.gz"
          bin_file="gum"
          ;;
linux-386) 
          download_url="https://api.github.com/repos/charmbracelet/gum/releases/assets/91458480" 
          download_file_name="gum_0.9.0_Linux_i386.tar.gz"
          bin_file="gum"
          ;;
linux-amd64) 
          download_url="https://api.github.com/repos/charmbracelet/gum/releases/assets/91458472" 
          download_file_name="gum_0.9.0_Linux_x86_64.tar.gz"
          bin_file="gum"
          ;;
windows-386) 
          download_url="https://api.github.com/repos/charmbracelet/gum/releases/assets/91458474" 
          download_file_name="gum_0.9.0_Windows_i386.zip"
          bin_file="gum"
          ;;
windows-amd64) 
          download_url="https://api.github.com/repos/charmbracelet/gum/releases/assets/91458473" 
          download_file_name="gum_0.9.0_Windows_x86_64.zip"
          bin_file="gum"
          ;;
      *)
      log_crit "platform $platform is not supported by $repo_url."
      exit 1
      ;;
    esac
  }

  #
  # untar: untar or unzip $1
  #
  # if you need to unpack in specific directory use a
  # subshell and cd
  #
  # (cd /foo && untar mytarball.gz)
  #
  untar() {
    tarball=$1
    case "${tarball}" in
      *.tar.gz | *.tgz | *.tar.xz | *.txz | *.tar.bz2 | *.tar.bz | *.tar) tar -xzf "${tarball}" ;;
      *.zip) unzip -qj "${tarball}" ;;
      *)
        log_crit "untar unknown archive format for ${tarball}"
        return 1
        ;;
    esac
  }

  
  mktmpdir() {
    TMPDIR="$(mktemp -d)"
    mkdir -p "${TMPDIR}"
    echo "${TMPDIR}"
  }

  abspath() {
    local old=$PWD
    cd $1
    local temp=$PWD
    cd $old
    echo $temp
  }
  
  start() {
    repo="charmbracelet/gum"
    # github repo such as "github.com/egoist/doko"
    repo_url="github.com/$repo"
  
    uname_os_check
    uname_arch_check
    platform_check

    mkdir -p "/tmp/tmp.tmp_gum"
    install_dir=$(abspath "/tmp/tmp.tmp_gum")
    bin_name="gum"
    github_token="$GITHUB_TOKEN"
    
  
    # API endpoint such as "http://localhost:3000"
    api="http://localhost:3000"

    # original_version such as "latest"
    original_version="latest"
  
    # version such as "master"
    version="v0.10.0"
    
    tmpdir="$(mktmpdir)"

    bin_dest="$tmpdir/$bin_name"
    tmp="$tmpdir/$download_file_name"
  
    echo
     
    if [ "$original_version" != "$version" ]; then
      log_info "Resolved version $original_version to $version"
    fi
    log_info "Downloading asset for $os $arch"

    if [ -n "$github_token" ]; then
      auth_header="Authorization: token $github_token"
    fi

    http_download_curl "$tmp" "$download_url" "accept: application/octet-stream" "$auth_header"

    
    

    # Extract
    cd "$tmpdir"
    untar "$tmp"

    # If the archive extracts to a directory with the same name, cd into it
    folder_name="$tmp"
    # Remove extension from the name
    folder_name="$(basename "$folder_name" .zip)"
    folder_name="$(basename "$folder_name" .tar.gz)"
    folder_name="$(basename "$folder_name" .tgz)"
    folder_name="$(basename "$folder_name" .tar.xz)"
    folder_name="$(basename "$folder_name" .txz)"
    folder_name="$(basename "$folder_name" .tar.bz2)"
    folder_name="$(basename "$folder_name" .tar.bz)"
    folder_name="$(basename "$folder_name" .tar)"
    
    if [ -d "$folder_name" ]; then
      cd "$folder_name"
    fi

    if [ -w "$install_dir" ]; then
      log_info "Installing $bin_name to $install_dir"
      install "$bin_file" "$install_dir/$bin_name"
    else
      log_warn "Permissions required for installation to $install_dir"
      sudo install "$bin_file" "$install_dir/$bin_name"
    fi
  
    log_info "Installation complete"
    echo
  }
  
  start  