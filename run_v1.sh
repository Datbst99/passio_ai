#!/usr/bin/env bash

sudo apt update
sudo apt install -y erlang

echo "deb https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-erlang/deb/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/rabbitmq-erlang.list
echo "deb https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-server/deb/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/rabbitmq-server.list

curl -fsSL https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-server/gpg.key | sudo apt-key add -
curl -fsSL https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-erlang/gpg.key | sudo apt-key add -

sudo apt install -y rabbitmq-server
sudo systemctl enable rabbitmq-server
sudo systemctl start rabbitmq-server
