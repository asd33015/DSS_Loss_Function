import os
def SaveFeature(features, batchImgName, SaveRoot, args, single=True):

    for feature, ImgName in zip(features, batchImgName):
        # with will automatically close the txt file
        tmp = ImgName.split('/')
        mp = tmp[4].split('.')
        type = tmp[3]
        subject = mp[0]
        SavePath = os.path.join(SaveRoot, type)
        if not os.path.isdir(SavePath): os.makedirs(SavePath)
        # FeaturePath = SavePath + '.txt'
        FeaturePath = SavePath + '/' + subject + '.txt'
        with open(FeaturePath, 'w') as f:
            for fea in feature:
                text = str(fea)
                f.write("{}\n".format(text))
