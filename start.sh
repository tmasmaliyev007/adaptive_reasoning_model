#!/bin/bash

# --- SSH ---
mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh

# Accept RunPod's injected public key
if [ -n "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# Allow root login with key (no password)
sed -i 's/#PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
sed -i 's/#PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# Inject environment variables for interactive SSH sessions only
env > /etc/environment

# Start sshd
/usr/sbin/sshd

# --- Nginx ---
nginx