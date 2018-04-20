
from common import *
from dataset.reader import *
from net.layer.box.process import *


def process_identity(instance):
    return instance


def process_norm_clip(instance, threshold=0.3, top=0.5):
    instance = (instance-instance.min())/(instance.max()-instance.min())
    instance = np.clip(instance,0,top)/top
    return instance

def process_clip(instance, top=0.4):
    instance = np.clip(instance,0,top)/top
    return instance

def process_norm_sq(instance):
    instance = (instance-instance.min())/(instance.max()-instance.min())
    instance = instance**0.5
    return instance


def process_sq(instance):
    instance = instance**0.5
    return instance

#ensemble =======================================================

class Cluster(object):
    def __init__(self):
        super(Cluster, self).__init__()
        self.members=[]
        self.center =[]
        #self.process_funct=process_identity
        self.process_funct=process_norm_clip

    def add_item(self, proposal, instance, ):
        instance = self.process_funct(instance)
        if self.center ==[]:
            self.members = [{
                'proposal': proposal, 'instance': instance
            },]
            self.center  = {
                'union_proposal': proposal, 'union_instance':instance,
                'inter_proposal': proposal, 'inter_instance':instance,
                'average_proposal': proposal, 'average_instance':instance,
            }
        else:
            N = len(self.members)
            self.members.append({
                'proposal': proposal, 'instance': instance
            })
            center_union_proposal = self.center['union_proposal'].copy()
            center_union_instance = self.center['union_instance'].copy()
            center_inter_proposal = self.center['inter_proposal'].copy()
            center_inter_instance = self.center['inter_instance'].copy()
            center_average_proposal = self.center['average_proposal'].copy()
            center_average_instance = self.center['average_instance'].copy()


            self.center['union_proposal'][1] = max(center_union_proposal[1],proposal[1])
            self.center['union_proposal'][2] = max(center_union_proposal[2],proposal[2])
            self.center['union_proposal'][3] = min(center_union_proposal[3],proposal[3])
            self.center['inter_proposal'][4] = min(center_union_proposal[4],proposal[4])
            self.center['union_proposal'][5] = min(center_union_proposal[5],proposal[5])
            self.center['union_instance'] = np.maximum(center_union_instance , instance )


            self.center['inter_proposal'][1] = max(center_inter_proposal[1],proposal[1])
            self.center['inter_proposal'][2] = max(center_inter_proposal[2],proposal[2])
            self.center['inter_proposal'][3] = min(center_inter_proposal[3],proposal[3])
            self.center['inter_proposal'][4] = min(center_inter_proposal[4],proposal[4])
            self.center['inter_proposal'][5] = min(center_inter_proposal[5],proposal[5])
            self.center['inter_instance'] = np.minimum(center_inter_instance , instance )

            self.center['inter_proposal'][1:6] = (N*center_average_proposal[1:6] + proposal[1:6]  )/(N+1)
            self.center['average_instance']    =  (N*center_average_instance + instance  )/(N+1)


    def compute_iou(self, proposal, instance, type='union'):

        if type=='union':
            center_proposal = self.center['union_proposal']
            center_instance = self.center['union_instance']
        elif type=='inter':
            center_proposal = self.center['inter_proposal']
            center_instance = self.center['inter_instance']
        elif type=='average':
            center_proposal = self.center['average_proposal']
            center_instance = self.center['average_instance']
        else:
            raise NotImplementedError

        x0 = int(max(proposal[1],center_proposal[1]))
        y0 = int(max(proposal[2],center_proposal[2]))
        x1 = int(min(proposal[3],center_proposal[3]))
        y1 = int(min(proposal[4],center_proposal[4]))

        w = max(0,x1-x0)
        h = max(0,y1-y0)
        box_intersection = w*h
        if box_intersection<0.01: return 0

        x0 = int(min(proposal[1],center_proposal[1]))
        y0 = int(min(proposal[2],center_proposal[2]))
        x1 = int(max(proposal[3],center_proposal[3]))
        y1 = int(max(proposal[4],center_proposal[4]))

        i0 = center_instance[y0:y1,x0:x1]>0.5  #center_inter[y0:y1,x0:x1]
        i1 = instance[y0:y1,x0:x1]>0.5

        intersection = np.minimum(i0, i1).sum()
        area    = np.maximum(i0, i1).sum()
        overlap = intersection/(area + 1e-12)


        if 0: #debug
            m = np.zeros((*instance.shape,3),np.uint8)
            m[:,:,0]=instance*255
            m[:,:,1]=center_instance*255

            cv2.rectangle(m, (x0,y0),(x1,y1),(255,255,255),1)
            image_show('m',m)
            print('%0.5f'%overlap)
            cv2.waitKey(0)

        return overlap

def sort_clsuters(clusters):

    value=[]
    for c in clusters:
        x0,y0,x1,y1 = (c.center['inter_proposal'] + c.center['union_proposal'])[1:5]
        value.append((x0+x1)+(y0+y1)*100000)
    value=np.array(value)
    index = list(np.argsort(value))

    return index



def do_clustering( proposals, instances, threshold=0.3, type='union'):

    clusters = []
    num_augments   = len(instances)
    for n in range(0, num_augments):
        proposal = proposals[n]
        instance = instances[n]

        num = len(instance)
        for i in range(num):
            p, m = proposal[i],instance[i]
            if m.sum()<5: continue

            max_iou=0
            if len(clusters)!=0:
                ious = np.array( [c.compute_iou(p, m, type) for c in clusters], np.float32 )
                k = np.argmax(ious)
                max_iou = ious[k]

                if max_iou>threshold:
                    c = clusters[k]
                    c.add_item(p, m)


            if max_iou<0.1:
                c = Cluster()
                c.add_item(p, m)
                clusters.append(c)
            #print(len(clusters),max_iou)

    return clusters






def run_make_l2_data():

    out_dir = \
        '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-05d/ensemble/data'

    ensemble_dirs = [
        '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-05d/predict/%s'%e

        for e in [
            'xx57_normal',
            'xx57_flip_transpose_1',
            'xx57_flip_transpose_2',
            'xx57_flip_transpose_3',
            'xx57_flip_transpose_4',
            'xx57_flip_transpose_5',
            'xx57_flip_transpose_6',
            'xx57_flip_transpose_7',
            #  57
            'xx57_scale_0.8',
            'xx57_scale_1.2',
            'xx57_scale_0.5',
            'xx57_scale_1.8',
            # 57
            'xx57_scale_0.8_flip_transpose_1',
            'xx57_scale_0.8_flip_transpose_2',
            'xx57_scale_0.8_flip_transpose_3',
            'xx57_scale_0.8_flip_transpose_4',
            'xx57_scale_0.8_flip_transpose_5',
            'xx57_scale_0.8_flip_transpose_6',
            'xx57_scale_0.8_flip_transpose_7',
            #  57
            'xx57_scale_1.2_flip_transpose_1',
            'xx57_scale_1.2_flip_transpose_2',
            'xx57_scale_1.2_flip_transpose_3',
            'xx57_scale_1.2_flip_transpose_4',
            'xx57_scale_1.2_flip_transpose_5',
            'xx57_scale_1.2_flip_transpose_6',
            'xx57_scale_1.2_flip_transpose_7',
            #  57
            'xx57_stretch_0.5_flip_transpose_1',
            'xx57_stretch_0.5_flip_transpose_2',
            'xx57_stretch_0.5_flip_transpose_3',
            'xx57_stretch_0.5_flip_transpose_4',
            'xx57_stretch_0.5_flip_transpose_5',
            'xx57_stretch_0.5_flip_transpose_6',
            'xx57_stretch_0.5_flip_transpose_7',

    ]]


    #setup ---------------------------------------
    os.makedirs(out_dir +'/ensemble_data_overlays', exist_ok=True)
    os.makedirs(out_dir +'/ensemble_data', exist_ok=True)


    # names = glob.glob(ensemble_dirs[0] + '/overlays/*/')
    # names = [n.split('/')[-2]for n in names]
    # sorted(names)

    split = 'test1_ids_gray2_53'  #'BBBC006'   #'valid1_ids_gray2_43' #
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')   #[:10] try 10 images for debug

    num_ensemble = len(ensemble_dirs)
    for i in range(len(ids)):
        folder, name = ids[i].split('/')[-2:]
        name = '1962d0c5faf3e85cda80e0578e0cb7aca50826d781620e5c1c4cc586bc69f81a'
        print('%05d %s'%(i,name))

        image = cv2.imread(DATA_DIR + '/image/%s/images/%s.png'%(folder,name), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]


        instances=[]
        proposals=[]
        for t,dir in enumerate(ensemble_dirs):
            instance = np.load(dir +'/instances/%s.npy'%(name))
            proposal = np.load(dir +'/detections/%s.npy'%(name))
            assert(len(proposal)==len(instance))
            #print(len(instance),len(proposal))

            instances.append(instance)
            proposals.append(proposal)

        clusters = do_clustering( proposals, instances, type='average', threshold=0.5)#union
        clusters = [clusters[i] for i in sort_clsuters(clusters)]
        num_clusters = len(clusters)
        print(num_clusters)

        # ensemble instance
        ensemble_instances      = []
        ensemble_instance_edges = []
        for k in range(num_clusters):
            c = clusters[k]
            num_members = len(c.members)

            ensemble_instance = np.zeros((height, width),np.float32)
            ensemble_instance_edge = np.zeros((height, width),np.float32)
            for j in range(num_members):
                m = c.members[j]['instance']

                kernel = np.ones((3,3),np.float32)
                me = m - cv2.erode(m,kernel)
                md = m - cv2.dilate(m,kernel)
                diff = ( me -md )*m

                ensemble_instance += m
                ensemble_instance_edge += diff


            #------------------------
            #np.percentile(a, 50)
            ensemble_instance = ensemble_instance #/num_members
            ensemble_instance_edge=ensemble_instance_edge #/num_members


            ensemble_instances.append(ensemble_instance)
            ensemble_instance_edges.append(ensemble_instance_edge)


            # image_show('ensemble_instance',ensemble_instance/ensemble_instance.max()*255,1)
            # image_show('ensemble_instance_edge',ensemble_instance_edge/ensemble_instance_edge.max()*255,1)
            # cv2.waitKey(0)

        ensemble_instances = np.array(ensemble_instances)
        ensemble_instance_edges = np.array(ensemble_instance_edges)

        sum_instance      = ensemble_instances.sum(axis=0)
        sum_instance_edge = ensemble_instance_edges.sum(axis=0)
        sum_instance_less_edge = np.clip(sum_instance-5*(sum_instance_edge>0.5),0,np.inf)

        gray0 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.uint8)
        gray1 = (sum_instance/sum_instance.max()*255).astype(np.uint8)
        gray2 = (sum_instance_edge/sum_instance_edge.max()*255).astype(np.uint8)
        all = np.hstack([gray0,gray1,gray2])
        image_show('all',all,1)

        #check clustering -----------------------------------------------------------------------------------
        image_accept = image.copy()
        image_reject = image.copy()
        for k in range(num_clusters):
            c = clusters[k]
            num_members = len(c.members)

            mean_instance     = ensemble_instances[k]/num_members
            variance_instance = np.zeros((height, width), np.float32)
            for j in range(num_members):
                m = c.members[j]['instance']
                variance_instance += (m-mean_instance)**2
            variance_instance = (variance_instance/num_members)**0.5

            xx=mean_instance*(1-variance_instance/variance_instance.max())


            score = variance_instance.sum()/mean_instance.sum()
            score = (mean_instance>0.5).sum()/(mean_instance>0.001).sum()

            #print(mean_instance.max())

            ensemble_instance = ensemble_instances[k]#/num_members
            ensemble_instance_edge = ensemble_instance_edges[k]#/num_members







            #binary = (mean_instance>0.5)
            binary = (xx>0.5)
            contour = mask_to_inner_contour(binary)

            #filter
            is_accept = 1
            area = binary.sum()

            if  area<16    : is_accept=0
            if  score<0.30 : is_accept=0



            image_show('inter_instance',c.center['inter_instance']*255,1)
            image_show('union_instance',c.center['union_instance']*255,1)
            image_show('ensemble_instance',ensemble_instance/ensemble_instance.max()*255,1)
            image_show('ensemble_instance_edge',ensemble_instance_edge/ensemble_instance_edge.max()*255,1)
            #image_show('mean_instance',mean_instance/mean_instance.max()*255,1)
            image_show('mean_instance',mean_instance/1*255,1)
            image_show('variance_instance',variance_instance/variance_instance.max()*255,1)

            image_show('sum_instance_less_edge',sum_instance_less_edge/sum_instance_less_edge.max()*255,1)
            image_show('xx',xx/xx.max()*255,1)
            xx_image = cv2.cvtColor((xx/xx.max()*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)


            #print(variance_instance.sum(),mean_instance.sum(),score)
            print( k, score)


            if is_accept:
                image_results = image_accept
                color = (0,255,0)
            else:
                image_results = image_accept #image_reject
                color = (0,0,255)


            image_results[contour] = color


            gray01 = cv2.cvtColor(gray0,cv2.COLOR_GRAY2BGR)
            gray11 = cv2.cvtColor(gray1,cv2.COLOR_GRAY2BGR)
            gray21 = cv2.cvtColor(gray2,cv2.COLOR_GRAY2BGR)
            image_show('image_results',np.hstack([image_accept,image_reject,xx_image, gray01,gray11,gray21]),1)
            cv2.waitKey(0)


        #save as train data
        #data  = cv2.merge((gray0, gray1, gray2))
        #cv2.imwrite( out_dir +'/ensemble_data_overlays/%s.png'%name, all)
        #cv2.imwrite( out_dir +'/ensemble_data/%s.png'%name, data)

        cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_make_l2_data()
    print('\nsucess!')