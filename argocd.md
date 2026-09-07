<!-- create namespace -->
kubectl create namespace argocd

<!-- install argocd -->

kubectl apply -n argocd \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml


<!-- port-forward and access --> 

kubectl port-forward svc/argocd-server -n argocd 8080:443