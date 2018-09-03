#!/bin/bash

sudo apt-get update
sudo apt-get install -y build-essential cmake git wget htop python3 python3-pip zlib1g-dev

if [ ! -d ~/deep-neuroevolution ]; then
	git clone https://github.com/uber-common/deep-neuroevolution.git
fi

if [ ! -f /etc/redis/redis.conf ]; then
	wget --quiet http://download.redis.io/releases/redis-3.2.7.tar.gz -O redis-3.2.7.tar.gz
	tar -xvzf redis-3.2.7.tar.gz
	cd redis-3.2.7
	make
	sudo make install
	sudo mkdir /etc/redis
	sudo cp ~/deep-neuroevolution/redis_config/redis_local_mirror.conf /etc/redis/redis.conf
	cd ..
	rm -rf redis-3.2.7 redis-3.2.7.tar.gz

	# Set up redis working directory
	sudo sed -ie 's/dir \.\//dir \/var\/lib\/redis/' /etc/redis/redis.conf
	sudo mkdir /var/lib/redis
	sudo chown ubuntu:ubuntu /var/lib/redis

	# rely on firewall for security
	sudo sed -ie "s/bind 127.0.0.1//" /etc/redis/redis.conf
	sudo sed -ie "s/protected-mode yes/protected-mode no/" /etc/redis/redis.conf

	# System settings for redis
	echo "vm.overcommit_memory=1" | sudo tee -a /etc/sysctl.conf
	sudo sysctl vm.overcommit_memory=1
	sudo apt-get install -y hugepages
	echo "sudo hugeadm --thp-never" | sudo tee /etc/profile.d/disable_thp.sh > /dev/null
	. /etc/profile.d/disable_thp.sh

	# Start redis with systemctl
	# sudo sed -ie "s/supervised no/supervised systemd/" /etc/redis/redis.conf
	# ^ doesn't seem to matter; if it's enabled, the logs show "systemd supervision requested, but NOTIFY_SOCKET not found"
	echo "
	[Unit]
	Description=Redis In-Memory Data Store
	After=network.target

	[Service]
	User=ubuntu
	Group=ubuntu
	ExecStart=/usr/local/bin/redis-server /etc/redis/redis.conf
	ExecStop=/usr/local/bin/redis-cli shutdown
	Restart=always

	[Install]
	WantedBy=multi-user.target
	" | sudo tee /etc/systemd/system/redis.service > /dev/null
	sudo systemctl start redis
fi

cd ~/deep-neuroevolution
sudo pip3 install -r requirements.txt
sudo pip3 install pillow==3.3.1

if [ ! -f ~/.ssh/id_rsa ]; then
	echo "-----BEGIN RSA PRIVATE KEY-----
MIIEpQIBAAKCAQEAo6d095z5XYSDRGUB7aX4m0jl2APWsxVSRuKLZoSs/H/gq+n+
RaQnydTbDukbetRaUMGy6IORnd1i+++rg9xiaEorYSbwRfjKtwRkJvd0zdu0B13I
+sQzzAJsPRNvMNjs3nnhrk0vsXuhllUVBn7M+90eJiaeRb8T+ubDnYyVirGnhN3u
llDZQ/DDi+aKZA/v1LhKEHcPSplTqg1FZqNBBNWOJk6HuiT/wPSZQepelcHZY8eX
CkRnBfRK99NRmLg1XgRBBDx7+KAsnkhF4D79RB1tlcRzm1RQV7HUELEoP8M8mbel
YbMe9ksZJ7/Mx0tfHaQmRmcOzCQpf9nLK6MYjQIDAQABAoIBAQCczpgxpZP9+jl4
sDu/xLbHu3qXl14B690RFIKzcU77BWB1+NftTJPfPPfEerEc6Rm8pUxSo7ZEB9uP
QJ8b0m/sM50LEq9IrFE4OZnpCFQ/51LBeChZtuNWh7/EabmxmTilFy2ZwFWBfs9e
ZxmlRpwMfFsl/PzMIYD4eGJYtFqZL4XAs+NvfyrODUBabRzvOc9TL46hWEP0hosW
YE8lnDJsDOHh7ArHdgdGag13sQ9r6XcJWNEaqZfqX3fIu3mKllTkJsUJlm5sPWrk
9/ANk7aFFbmz1Ncz4kYGuOSyDeDiSFjDYBZhg0nXhcP9eqAb7BVpVFdSwGpy6W8N
SXZYYp4hAoGBANMLR9uX0TCSuMAYlWKgoEAiXUz1DKqoX9BZQpztl7NSiOVyZryC
9WlBty+H275XZJDLAGz/13QUZcByXU1G0va/qDLDG6bYq9WoogTot5q2sMhJKd4Z
xRZOi68gDQiVCfE0DydZJuYcB1bb2fyFOxnkAAitWx/qpUf/9Q4Kq1H5AoGBAMaD
5Mu7GF1rCYU+Vm/1FHqVsCpHKN28ELCb06pksEAT/VNYKp3+fsrN2cYATCoFln0l
n1YnQwA5zbSrjJGBhECkjKuVS4pkGzPvP+YGSq+MAj5BxQoKlNlSpSZgD4WldbOh
/Z/JV9Fn2At8SlcCTFcPLbc78VPP1vSNKUhmTCA1AoGBAK8l748JYi1Dt1yFioT9
9cEERBZ4UPjZIBuT2LrQXFQQrVhvJ2BP90hRp6wkvnQrp2SbdVEAy1ilDQU4ZMKb
gr4RtY/baPmBXKrHdx9H3AjkkbbHMZ4IGQ84RKkkmmyC9Gtf3yuyy5uxq02kzDbM
g44rMPQCm1vTqzQj8saiiChRAoGBAJ+DWfSO6Twfhy3m8mPSBduerki6l07dEHgp
LoLrl2hV56fx34TG+7EQid39XTYi+VKkSY1bzQ3AZNe5RSGiddfPoS06sNGKMQWh
SLIX+ilnEmJeTOsNp5+dNhFI/RAB7Tsjfn3HtuYQUFyiScXylc8a5jwnUXpsNqiZ
SUPg12jFAoGALR7JFQTAKP6PL/vHgXwbRaOLXK461HM483APCnWe9jDnvSgJ4+KP
TACWVTFmwFuL+/oTyUvo49MttX4Xkgk8yja9N1CeNkRuVKeogY5fmrTWuT2l31DK
bbYI9zNekAiQx1p8PLuiDK8m1AmLnhJw22uM3tjhdWAMCgTAYgLXjWM=
-----END RSA PRIVATE KEY-----" | tee ~/.ssh/id_rsa > /dev/null

	chmod 600 ~/.ssh/id_rsa

	echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCjp3T3nPldhINEZQHtpfibSOXYA9azFVJG4otmhKz8f+Cr6f5FpCfJ1NsO6Rt61FpQwbLog5Gd3WL776uD3GJoSithJvBF+Mq3BGQm93TN27QHXcj6xDPMAmw9E28w2OzeeeGuTS+xe6GWVRUGfsz73R4mJp5FvxP65sOdjJWKsaeE3e6WUNlD8MOL5opkD+/UuEoQdw9KmVOqDUVmo0EE1Y4mToe6JP/A9JlB6l6Vwdljx5cKRGcF9Er301GYuDVeBEEEPHv4oCyeSEXgPv1EHW2VxHObVFBXsdQQsSg/wzyZt6Vhsx72Sxknv8zHS18dpCZGZw7MJCl/2csroxiN ubuntu@ip-172-31-27-159" | tee ~/.ssh/id_rsa.pub > /dev/null

	chmod 600 ~/.ssh/id_rsa.pub
fi

if [ ! -f /mnt/2Gib.swap ]; then
	sudo fallocate -l 2g /mnt/2GiB.swap
	sudo dd if=/dev/zero of=/mnt/2GiB.swap bs=1024 count=2097152
	sudo chmod 600 /mnt/2GiB.swap
	sudo mkswap /mnt/2GiB.swap
	sudo swapon /mnt/2GiB.swap
	echo '/mnt/2GiB.swap swap swap defaults 0 0' | sudo tee -a /etc/fstab > /dev/null
fi


screen -d -m ssh -o StrictHostKeyChecking=no -L 6379:10.133.20.240:6379 root@174.138.0.49

cd ~/deep-neuroevolution

screen -d -m python3 -m es_distributed.main workers --master_host localhost --relay_socket_path /tmp/es_redis_relay.sock --algo 'es' --num_workers 4