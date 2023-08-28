import torch
import logging
import argparse
import torchvision
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import models
import torch.utils.data as data
from torchvision import transforms

def get_args():
        parser = argparse.ArgumentParser(description='Modelos pre-entrenados con pytorch')
        parser.add_argument('--epochs', '-e', type=int, default=4, help='Numero de epocas')
        parser.add_argument('--model', '-m', type=str, default='resnet50', help='Modelo a utilizar')
        parser.add_argument('--batch_size', '-bs', type=int, default=32, help='Tamaño del batch')
        parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='Learning rate', dest='lr')
        parser.add_argument('--re_dim', '-rd', type=int, default=120, help='Re dimensionar las imagenes')
        parser.add_argument('--clases', '-c', type=int, default=2, help='Numero de clases')
        parser.add_argument('--neurons', '-n', type=int, default=500, help='Numero de capas')
        parser.add_argument('--num_cap', '-nc', type=int, default=1, help='Numero de capas')
        parser.add_argument('--ent_path', '-ep', type=str,
                            default= 'D:\\rafaortizferegrino\\Bases de datos\\AumentoDatos\\ent\\',
                            help='Direccion de los datos de entrenamiento')
        parser.add_argument('--val_path', '-cvp', type=str,
                            default= 'D:\\rafaortizferegrino\\Bases de datos\\AumentoDatos\\val\\',
                            help='Direccion de los datos de validacion')

        return parser.parse_args()

def creatingData(args):
        trainDataPath = args.ent_path
        valDataPath = args.val_path

        transforms_images = transforms.Compose([
                transforms.Resize((args.re_dim, args.re_dim)),
                transforms.ToTensor(),
                transforms.Normalize(mean= [0.485, 0.456, 0.406],
                            std= [0.229, 0.224, 0.225])
                ])

        valData = torchvision.datasets.ImageFolder(root = valDataPath, transform = transforms_images)
        trainData = torchvision.datasets.ImageFolder(root= trainDataPath, transform= transforms_images)

        trainDataLoader= data.DataLoader(trainData, batch_size= args.batch_size, shuffle=True)
        valDataLoader= data.DataLoader(valData, batch_size= args.batch_size, shuffle=True)

        return trainDataLoader, valDataLoader

def checkingGPU():
        if torch.cuda.is_available():
                device = torch.device("cuda")
        else:
                device = torch.device("cpu")
        logging.info(f'Using device {device}')
        return device


def getNeurons(nc):
        ln = []
        for i in range(nc):
                ln.append(int(input(f'Layer {i} neurons: ')))
        return ln

def trainingModel(model, output= 2, n_layers= 1, n_features= []):
        transfer_model = model
        layers = []
        if n_layers > 1:
                layers.append(nn.Linear(transfer_model.fc.in_features, n_features[0]))
                for i in range(len(n_features)- 1):
                        layers.append(nn.Linear(n_features[i], n_features[i + 1]))
        else:
                layers.append(nn.Linear(transfer_model.fc.in_features, n_features[0]))

        transfer_model.fc = nn.Sequential(*layers,
                                          nn.ReLU(),
                                          nn.Dropout(),
                                          nn.Linear(n_features[-1], output),
                                          nn.Softmax(dim= 0))
        return transfer_model

def menuDisplay(modelv):
        preTrained = {'resnet50': models.resnet50,
                      'resnet101': models.resnet101,
                      'resnet152': models.resnet152,
                      'squeezenet1_0': models.squeezenet1_0,
                      'squeezenet1_1': models.squeezenet1_1,
                      'convnext_large': models.convnext_large,
                      'convnext_small': models.convnext_small,
                      'inception_v3': models.inception_v3,
                      'mobilenet_v3_large': models.inception_v3,
                      'googlenet': models.googlenet,
                      'efficientnet_b7': models.efficientnet_b7,
                      'efficientnet_b0': models.efficientnet_b0,
                      }

        if modelv in preTrained.keys():
                return preTrained[modelv](weights='DEFAULT')
        else:
                logging.error(f'No model found for {modelv}')
                return None

def train(model, optimizer, lossFn, trainLoader, valLoader, epochs, device, modelName):
        best_model = 1000
        for epoch in range(epochs):

                trainingLoss = 0.0
                trainingAcc = 0.0
                totalt = 0
                model.train()

                for batch in tqdm(trainLoader, ncols= 70, desc= 'Training'):
                        optimizer.zero_grad()

                        inputs, targets = batch
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                        output = model(inputs)
                        loss = lossFn(output, targets)

                        loss.backward()
                        optimizer.step()

                        trainingLoss += loss.item()
                        _, predicted = torch.max(output.data, 1)
                        trainingAcc += (predicted == targets).sum().item()
                        totalt += targets.size(0)

                valLoss = 0.0
                valAcc = 0.0
                totalv = 0
                model.eval()

                with torch.no_grad():
                        for batch in tqdm(valLoader, ncols= 70, desc= 'Validating'):

                                inputs, targets = batch
                                inputs = inputs.to(device)
                                targets = targets.to(device)

                                output = model(inputs)
                                loss = lossFn(output, targets)

                                valLoss += loss.item()
                                _, predicted = torch.max(output.data, 1)
                                valAcc += (predicted == targets).sum().item()
                                totalv += targets.size(0)

                if best_model > valAcc/totalv:
                        best_model = valAcc/totalv
                        torch.save(best_model, modelName + '_' + str(epoch) + '.pth')

                print(f'Epoca: {epoch}, train_loss: {(trainingLoss/totalt):.4f}, train_acc: {(trainingAcc/totalt):.4f}')
                print(f'Epoca: {epoch}, val_loss: {(valLoss/totalv):.4f}, val_acc: {(valAcc/totalv):.4f}')
                print("------------------------------------")

def main():
        args = get_args()
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        device = checkingGPU()
        logging.info(f'''Starting training:
        Model:            {args.model}
        N_Dense:          {args.num_cap}
        Epochs:           {args.epochs}
        Batch size:       {args.batch_size}
        Learning rate:    {args.lr}
        Images dimension: {args.re_dim}
        Device:           {device}
        ''')
        n_features= getNeurons(args.num_cap) if args.num_cap > 1 else [args.neurons]
        trainData, valData = creatingData(args)
        model = trainingModel(menuDisplay(args.model), output = args.clases,
                              n_layers= args.num_cap,
                              n_features= n_features)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr= args.lr, momentum=0.9)
        loss= torch.nn.CrossEntropyLoss()

        try:
                train(model, optimizer, loss, trainData, valData, args.epochs, device, args.model)
        except torch.cuda.OutOfMemoryError:
                logging.error('Detected OutOfMemoryError! '
                        'Enabling checkpointing to reduce memory usage, but this slows down training. '
                        'Consider enabling AMP (--amp) for fast and memory efficient training')
                torch.cuda.empty_cache()

if __name__ == '__main__':
     main()