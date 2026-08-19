worker_processes __NGINX_WORKERS__;
user root;
error_log __ROOT__/logs/nginx-error.log warn;
pid __ROOT__/run/nginx.pid;

events {
    worker_connections 2048;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    sendfile on;
    keepalive_timeout 65;
    client_max_body_size 80m;
    access_log off;

    map $http_upgrade $connection_upgrade {
        default upgrade;
        '' close;
    }

    server {
        listen __LISTEN_PORT__ ssl;
        server_name _;

        ssl_certificate     __ROOT__/proxy/certs/server.crt;
        ssl_certificate_key __ROOT__/proxy/certs/server.key;
        ssl_protocols TLSv1.2 TLSv1.3;

        proxy_http_version 1.1;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Forwarded-Host $http_host;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;

        location /avatar/ {
            alias __ROOT__/assets/;
            add_header Cache-Control "public, max-age=86400";
        }

        location /av/ {
            proxy_pass http://127.0.0.1:__AVATAR_GW_PORT__/;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;
            proxy_buffering off;
            proxy_cache off;
            proxy_read_timeout 3600s;
            proxy_send_timeout 3600s;
            chunked_transfer_encoding on;
            client_max_body_size 80m;
        }

        location /v1/realtime {
            proxy_pass http://127.0.0.1:__S2S_PORT__/v1/realtime;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;
        }

        location / {
            proxy_pass http://127.0.0.1:__WEB_PORT__;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;
        }
    }
}
