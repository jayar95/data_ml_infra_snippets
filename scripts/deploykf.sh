apt upgrade -y
apt install apt-transport-https ca-certificates curl software-properties-common git -y
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
curl -sfL https://get.k3s.io | sh -
git clone -b main https://github.com/xdimensionio/deployKF ./deploykf
chmod +x ./deploykf/argocd-plugin/install_argocd.sh
bash ./deploykf/argocd-plugin/install_argocd.sh

echo "x" | base64 -d > gh-repo-creds.yaml

kubectl apply -f gh-repo-creds.yaml

VERSION=$(curl -L -s https://raw.githubusercontent.com/argoproj/argo-cd/stable/VERSION)
curl -sSL -o argocd-linux-amd64 https://github.com/argoproj/argo-cd/releases/download/v$VERSION/argocd-linux-amd64
sudo install -m 555 argocd-linux-amd64 /usr/local/bin/argocd
rm argocd-linux-amd64

export KUBECONFIG="/etc/rancher/k3s/k3s.yaml"
kubectl config set-context --current --namespace=argocd

argocd login --core

argocd app sync deploykf --resource 'argoproj.io:Application:cert-manager'
argocd app sync deploykf --resource ':Secret:cloudflare-credentials'
argocd app sync deploykf --resource 'cert-manager.io:ClusterIssuer:letsencrypt-prod'