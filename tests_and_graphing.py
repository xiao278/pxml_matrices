import numpy as np

def cosine_similarity(A, k):
    '''
    Takes a matrix A and a rank k to use

    Returns the cosine similarity matrix of the rank-k approximation of A
    '''
    CM = np.zeros(A.shape)
    U, S, V = np.linalg.svd(A)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    V_k = V[:k, :]

    for i, u in enumerate(U_k @ S_k):
        for j, v in enumerate((S_k @ V_k).T):
            CM[i,j] = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
    return CM



def community_detection(A, threshold):
    '''
    Finds the community blocks looking at where ones are adjacent

    returns a tensor of indices where the second index is the first index not included

    Only works for symmetric square matrices right now, which I think all cosine matrices are
    '''
    communities = []
    for i, row in enumerate(A):
        if len(communities) != 0 and communities[-1][1] > i: continue #If there is an index in the communities that is larger than i
        first_one = i 
        for j in range(i, len(row)+1):
            if j == len(row) or A[i,j] < threshold: # If all values past i meet the threshhold or if there is a value down the line that doesn't
                last_one = j
                communities.append((i,j))
                break
    
    for i in range(len(communities)): communities[i] = slice(communities[i][0], communities[i][1]) #Turns tuples into slices

    return communities
        

def community_strength(A, threshold):
    '''
    takes the cosine similarity matrix and checks how stong each community is using an error
    '''
    communities = community_detection(A, threshold)
    strengths = []
    for community_slice in communities:
        community = A[community_slice, community_slice]

        delta = community - np.ones(community.shape)
        print(delta)
        strengths.append(1 - np.round(np.linalg.norm(delta, ord=2), decimals = 3))

    return strengths

