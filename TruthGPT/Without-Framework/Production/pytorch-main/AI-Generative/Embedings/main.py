# Great example of how to use the embedding layer in pytorch
# https://github.com/chonkie-inc/chonkie/tree/main/src/chonkie/embeddings

# More faster: pip install chonkie (limited)
# Research 
# https://github.com/ivana-13/guided_masking/blob/main/volta/embeddings.py 

# # resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377
#    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None)
