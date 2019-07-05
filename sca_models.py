import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    shift to only output the feature map
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        feature_map = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        #out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        #out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return feature_map

    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Spatial_attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self,feature_map,decoder_dim,K = 512):
        """
        :param feature_map: feature map in level L
        :param decoder_dim: size of decoder's RNN
        """
        super(Spatial_attention, self).__init__()
        _,C,H,W = tuple([int(x) for x in feature_map])
        self.W_s = nn.Parameter(torch.randn(C,K))
        self.W_hs = nn.Parameter(torch.randn(K,decoder_dim))
        self.W_i = nn.Parameter(torch.randn(K,1))
        self.bs = nn.Parameter(torch.randn(K))
        self.bi = nn.Parameter(torch.randn(1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 0)  # softmax layer to calculate weights
        
    def forward(self, feature_map, decoder_hidden):
        """
        Forward propagation.

        :param feature_map: feature map in level L(batch_size, C, H, W)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: alpha
        """
        V_map = feature_map.view(feature_map.shape[0],2048,-1) 
        V_map = V_map.permute(0,2,1)#(batch_size,W*H,C)
        # print(V_map.shape)
        # print("m1",torch.matmul(V_map,self.W_s).shape)
        # print("m2",torch.matmul(decoder_hidden,self.W_hs).shape)
        att = self.tanh((torch.matmul(V_map,self.W_s)+self.bs) + (torch.matmul(decoder_hidden,self.W_hs).unsqueeze(1)))#(batch_size,W*H,C)
        # print("att",att.shape)
        alpha = self.softmax(torch.matmul(att,self.W_i) + self.bi)
#         print("alpha",alpha.shape)
        alpha = alpha.squeeze(2)
        feature_map = feature_map.view(feature_map.shape[0],2048,-1) 
        # print("feature_map",feature_map.shape)
        # print("alpha",alpha.shape)
        temp_alpha = alpha.unsqueeze(1)
        attention_weighted_encoding = torch.mul(feature_map,temp_alpha)
        return attention_weighted_encoding,alpha


class Channel_wise_attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self,feature_map,decoder_dim,K = 512):
        """
        :param feature_map: feature map in level L
        :param decoder_dim: size of decoder's RNN
        """
        super(Channel_wise_attention, self).__init__()
        _,C,H,W = tuple([int(x) for x in feature_map])
        self.W_c = nn.Parameter(torch.randn(1,K))
        self.W_hc = nn.Parameter(torch.randn(K,decoder_dim))
        self.W_i_hat = nn.Parameter(torch.randn(K,1))
        self.bc = nn.Parameter(torch.randn(K))
        self.bi_hat = nn.Parameter(torch.randn(1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 0)  # softmax layer to calculate weights
        
    def forward(self, feature_map, decoder_hidden):
        """
        Forward propagation.

        :param feature_map: feature map in level L(batch_size, C, H, W)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: alpha
        """
        V_map = feature_map.view(feature_map.shape[0],2048,-1) .mean(dim=2)
        V_map = V_map.unsqueeze(2)#(batch_size,C,1)
        # print(feature_map.shape)
        # print(V_map.shape)
        # print("wc",self.W_c.shape)
        # print("whc",self.W_hc.shape)
        # print("decoder_hidden",decoder_hidden.shape)
        # print("m1",torch.matmul(V_map,self.W_c).shape)
        # print("m2",torch.matmul(decoder_hidden,self.W_hc).shape)
        # print("bc",self.bc.shape)
        att = self.tanh((torch.matmul(V_map,self.W_c) + self.bc) + (torch.matmul(decoder_hidden,self.W_hc).unsqueeze(1)))#(batch_size,C,K)
#         print("att",att.shape)
        beta = self.softmax(torch.matmul(att,self.W_i_hat) + self.bi_hat)
        beta = beta.unsqueeze(2)
        # print("beta",beta.shape)
        attention_weighted_encoding = torch.mul(feature_map,beta)

        return attention_weighted_encoding,beta


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    shift to sca attention
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,encoder_out_shape=[1,2048,8,8], K=512,encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.Spatial_attention = Spatial_attention(encoder_out_shape, decoder_dim, K)  # attention network
        self.Channel_wise_attention = Channel_wise_attention(encoder_out_shape, decoder_dim, K) # ATTENTION 
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        self.AvgPool = nn.AvgPool2d(8)
    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = self.AvgPool(encoder_out).squeeze(-1).squeeze(-1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        # encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        # num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)#需要更改形状？
        #alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)#需要更改形状

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            # attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
            #                                                     h[:batch_size_t])
            #channel-spatial模式attention
            #channel_wise
            attention_weighted_encoding, beta = self.Channel_wise_attention(encoder_out[:batch_size_t],h[:batch_size_t])
            #spatial
            attention_weighted_encoding, alpha = self.Spatial_attention(attention_weighted_encoding[:batch_size_t],h[:batch_size_t])
            #对attention_weighted_encoding降维
            attention_weighted_encoding = attention_weighted_encoding.view(attention_weighted_encoding.shape[0],2048,8,8)
            attention_weighted_encoding = self.AvgPool(attention_weighted_encoding)
            attention_weighted_encoding = attention_weighted_encoding.squeeze(-1).squeeze(-1)
            # gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            # attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            #alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, sort_ind
