Node = collections.namedtuple("Node", 'data axis index left right')
class KDTree(object):
     ## this code this wiki page as reference : https://en.wikipedia.org/wiki/K-d_tree
    def __init__(self, k, feat):
        def build_tree(feat, axis=0):
            if not feat:
                return None
            feat.sort(key=lambda x: x[0][axis])
            median = len(feat) // 2
            data, index = feat[median]
            next_axis = (axis + 1) % k                        
            return Node(data,axis,index,build_tree(feat[:median], next_axis),build_tree(feat[median + 1:], next_axis))
        self.root = build_tree(list(feat))
               
    def NN(self, input1):
        res = [None, None, float('inf'),float('inf')]
        ## res[2] = distance, res[3] = second smallest distance
        def search_NN(search_value):
            if search_value is None:
                return
            data, axis, index, left, right = search_value
            distance1 = 0
            for i, j in zip(data,input1):
                distance1 += math.sqrt((i-j)*(i-j))
            if distance1 < res[2]:
                res[0] = data
                res[1] = index
                res[2] = distance1
                res[3] = res[2]
            elif distance1 < res[3]:
                res[3] = distance1
                ## if found a closer point, then update to this point and update index
            diff = input1[axis] - data[axis]
            if diff <= 0:
                close = left
                away = right
            else:
                close = right
                away = left
            ## if this point's value on this axis is less than the node, 
            ## then we will need to search the left subtree, right otherwise
            search_NN(close)
            ## if this point's distance to its previous NN is bigger than this point's distance to boundary,still need to search the right subtree.
        search_NN(self.root)
        return res[0], res[1],res[2]/res[3]

def match_features(feats0, feats1, scores0, scores1, mode='naive'):
   ##########################################################################
   # TODO: YOUR CODE HERE
   ## raise NotImplementedError('match_features')
    if mode =='naive':
        ## use NNDR formula on Page 236
        matches = np.zeros(feats0.shape[0],dtype= int)
        scores = np.zeros(feats0.shape[0])
        for i in range(feats0.shape[0]):
            d1 = float('Inf')
            d2 = float('Inf')
            res = -1
            for j in range(feats1.shape[0]):
                d = np.linalg.norm(feats0[i]-feats1[j])
                if d < d1:
                    d2 = d1
                    d1 = d
                    res = j
                elif d < d2:
                    d2 = d
            r = d1/d2
            ## each feature of feat0 have a r score
            matches[i] = int(res)
            scores[i] = r

    if mode == 'kdtree':
          k = 50
          points = [(tuple(feats0[i]), i) for i in range(len(feats0))]
          tree = KDTree(k, points)
          ## put feats 0 into kdtree
          input1 = [feats1[i] for i in range(len(feats1))]
          ## search best matches in feats1 amoung feats0
          matches = []
          scores = []
          for ele in input1:
              matches.append(tree.NN(ele)[1])
              scores.append(tree.NN(ele)[2])
          matches = np.array(matches)
          scores = np.array(scores)
    
   ##########################################################################
    return matches, scores