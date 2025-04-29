import numpy as np


def cosine_similarity(A, k):
    '''
    Takes a matrix A and a rank k to use

    Returns the cosine similarity matrix of the rank-k approximation of A
    '''
    CM = np.zeros(A.shape)
    U, S, V = np.linalg.svd(A)
    U_k = U[:, :k]
    S_k = S[:k]
    V_k = V[:k, :]

    for i, u in enumerate(U_k):
        for j, v in enumerate((V_k).T):
            CM[i,j] = np.dot(u, v)/ (np.linalg.norm(u) * np.linalg.norm(v))
    
    return CM



def strict_community_detection(A, threshold = .9, sliced = True):
    '''
    Finds the community blocks looking at where ones are adjacent

    returns a tensor of indices where the second index is the first index not included

    Only works for symmetric square matrices right now, which I think all cosine matrices are
    '''

    threshold = 0.9
    communities = []
    for i, row in enumerate(A):
        if len(communities) != 0 and communities[-1][1] > i: continue #If there is an index in the communities that is larger than i
        first_one = i 
        for j in range(i+1, len(row)+1):
            if j == len(row) or A[i,j] < threshold: # If all values past i meet the threshold or if there is a value down the line that doesn't
                last_one = j
                communities.append((i,j))
                break
    
    if sliced:
        for i in range(len(communities)): communities[i] = slice(communities[i][0], communities[i][1]) #Turns tuples into slices
    
    return communities



def community_strengths(A, threshold, order = 'fro'):
    '''
    takes the cosine similarity matrix and checks how stong each community is using an error
    '''
    communities = community_detection(A, threshold)
    strengths = []
    for community_slice in communities:
        community = A[community_slice, community_slice]
        strengths.append(community_strength(community, order))
    return strengths

def community_strength(community_block, order):
    '''
    Takes a community block given by the reordered cosine matrix and judges how fit it is.
    Delta is the matrix of ones minus the community block
    We take the norm of delta and divide by 2k where k is how many nodes are in the block
    -1 <= 1 - ||Delta||/k <= 1
    '''
    return 1 - np.linalg.norm((np.ones(community_block.shape) - community_block), ord = order)/( len(community_block))




def community_detection(A, threshold, sliced = True):
    '''
    If the size of a community is 1 then check if it can belong in the other communities
    I can do this be
    '''
    
    #for i, community in enumerate(communities)
       # if community[1] - community[0] = 1
    return strict_community_detection(A, threshold=threshold, sliced = sliced)
"""
TODO: 
- call community_detection "strict_community_detection" and set the threshold to a constant of .9
- create a new community_detection function that has a threshold as input and tries to consolidate communities
    produced from the strict_community_detection
- consider other approaches to community_strength function
- begin applying community_detection to larger data
"""