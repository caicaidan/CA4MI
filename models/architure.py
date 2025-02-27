import torch
from .backbones import EEGNet, DeepConvNet as DCN, ShallowConvNet as SCN

class Cls_HEAD(torch.nn.Module):
    """
    Classification head with shared and private encoders for each subject.

    Args:
        args (Namespace): Configuration with attributes like input size, number of subjects,
                          latent dimensions, and hidden layer sizes.
    """
    def __init__(self, args):
        super(Cls_HEAD, self).__init__()
        ncha, size, _ = args.inputsize
        self.sub_cls = [[s, args.n_classes] for s in range(args.n_subjects)]
        self.latent_dim = args.latent_dim
        self.num_subjects = args.n_subjects
        self.hidden1 = args.hidden_dim[0]
        self.hidden2 = args.hidden_dim[1]
        self.samples = args.n_samples

        # Shared and private feature extractors
        self.shared = Shared(args)
        self.private = Private(args)

        self.head = torch.nn.ModuleList()
        for i in range(self.num_subjects):
            self.head.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.latent_dim*2 , self.hidden1),
                    torch.nn.ELU(inplace=True),
                    torch.nn.Dropout(),
                    torch.nn.Linear(self.hidden1, self.hidden2),
                    torch.nn.ELU(inplace=True),
                    torch.nn.Linear(self.hidden2, self.sub_cls[i][1])
                ))


    def forward(self, x_s, x_p, sub_module_label, sub_id):
        x_s = self.shared(x_s)
        x_p = self.private(x_p, sub_id)
        x = torch.cat([x_p, x_s], dim=1)
        return torch.stack([self.head[sub_module_label[i]].forward(x[i]) for i in range(x.size(0))])

    def get_encoded(self, x_s, x_p, sub_id):
        return self.shared(x_s), self.private(x_p, sub_id)

class Private(torch.nn.Module):
    def __init__(self, args):
        """
        Private encoder for each subject.

        Args:
            args (Namespace): Configuration with model type and subject count.
        """
        super(Private, self).__init__()
        self.out = torch.nn.ModuleList([self._get_model(args) for _ in range(args.n_subjects)])

    def _get_model(self, args):
        """Helper function to initialize model based on type."""
        if args.model == 'EEGNet':
            return EEGNet(args)
        elif args.model == 'SCN':
            return SCN(args)
        elif args.model == 'DCN':
            return DCN(args, latent_dim=args.latent_dim, kernel_1=64, kernel_2=16, dropout=args.dropout,
                       block_out_channels=[25, 25, 50, 100, 200])
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

    def forward(self, x_p, sub_id):
        """
        Forward pass for the private encoder of a specific subject.

        Args:
            x_p (Tensor): Private input data.
            sub_id (int): Subject ID for selecting the specific model.

        Returns:
            Tensor: Flattened output from the private encoder.
        """
        x_p = self.out[sub_id](x_p)
        return x_p.view(x_p.size(0), -1)

class Shared(torch.nn.Module):
    def __init__(self, args):
        """
        Shared encoder used across all subjects.

        Args:
            args (Namespace): Configuration with model type.
        """
        super(Shared, self).__init__()
        self.model = self._get_model(args)

    def _get_model(self, args):
        """Helper function to initialize shared model based on type."""
        if args.model == 'EEGNet':
            return EEGNet(args)
        elif args.model == 'SCN':
            return SCN(args)
        elif args.model == 'DCN':
            return DCN(args, latent_dim=args.latent_dim, kernel_1=64, kernel_2=16, dropout=args.dropout,
                       block_out_channels=[25, 25, 50, 100, 200])
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

    def forward(self, x_s):
        """
        Forward pass for the shared encoder.

        Args:
            x_s (Tensor): Shared input data.

        Returns:
            Tensor: Output from the shared encoder.
        """
        return self.model(x_s)
