rm -f assignment1.zip 
zip -r assignment1_fubaowu.zip . -x "*.git*" "*datasets/cifar-10-batches-py*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements.txt"
