Search.setIndex({docnames:["hydra","index","modules","quickstart","sesemi","sesemi.config","sesemi.models","sesemi.models.backbones","sesemi.models.heads","sesemi.ops","sesemi.ops.conf","sesemi.schedulers","sesemi.trainer","sesemi.trainer.conf"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["hydra.rst","index.rst","modules.rst","quickstart.rst","sesemi.rst","sesemi.config.rst","sesemi.models.rst","sesemi.models.backbones.rst","sesemi.models.heads.rst","sesemi.ops.rst","sesemi.ops.conf.rst","sesemi.schedulers.rst","sesemi.trainer.rst","sesemi.trainer.conf.rst"],objects:{"":{sesemi:[4,0,0,"-"]},"sesemi.collation":{JigsawTransformer:[4,1,1,""],RotationTransformer:[4,1,1,""]},"sesemi.collation.JigsawTransformer":{crop_patch:[4,2,1,""],transform:[4,2,1,""]},"sesemi.config":{resolvers:[5,0,0,"-"],structs:[5,0,0,"-"]},"sesemi.config.resolvers":{AttributeResolver:[5,1,1,""],SESEMIConfigAttributes:[5,1,1,""]},"sesemi.config.resolvers.SESEMIConfigAttributes":{iterations_per_epoch:[5,3,1,"id0"],max_iterations:[5,3,1,"id1"],num_gpus:[5,3,1,"id2"],num_nodes:[5,3,1,"id3"]},"sesemi.config.structs":{ClassifierConfig:[5,1,1,""],ClassifierHParams:[5,1,1,""],ClassifierModelConfig:[5,1,1,""],DataConfig:[5,1,1,""],DataLoaderConfig:[5,1,1,""],DatasetConfig:[5,1,1,""],EMAConfig:[5,1,1,""],LRSchedulerConfig:[5,1,1,""],LearnerConfig:[5,1,1,""],LossCallableConfig:[5,1,1,""],LossHeadConfig:[5,1,1,""],RunConfig:[5,1,1,""],RunMode:[5,1,1,""],SESEMIBaseConfig:[5,1,1,""],SESEMIInferenceConfig:[5,1,1,""],SESEMIPseudoDatasetConfig:[5,1,1,""]},"sesemi.config.structs.ClassifierConfig":{hparams:[5,3,1,"id4"]},"sesemi.config.structs.ClassifierHParams":{lr_scheduler:[5,3,1,"id5"],model:[5,3,1,"id6"],num_classes:[5,3,1,"id7"],optimizer:[5,3,1,"id8"]},"sesemi.config.structs.ClassifierModelConfig":{backbone:[5,3,1,"id9"],ema:[5,3,1,"id10"],regularization_loss_heads:[5,3,1,"id11"],supervised_loss:[5,3,1,"id12"]},"sesemi.config.structs.DataConfig":{test:[5,3,1,"id13"],train:[5,3,1,"id14"],val:[5,3,1,"id15"]},"sesemi.config.structs.DataLoaderConfig":{batch_sampler:[5,3,1,"id16"],batch_size:[5,3,1,"id17"],batch_size_per_gpu:[5,3,1,"id18"],collate_fn:[5,3,1,"id19"],dataset:[5,3,1,"id20"],drop_last:[5,3,1,"id21"],num_workers:[5,3,1,"id22"],pin_memory:[5,3,1,"id23"],sampler:[5,3,1,"id24"],shuffle:[5,3,1,"id25"],timeout:[5,3,1,"id26"],worker_init_fn:[5,3,1,"id27"]},"sesemi.config.structs.DatasetConfig":{image_transform:[5,3,1,"id28"],name:[5,3,1,"id29"],root:[5,3,1,"id30"],subset:[5,3,1,"id31"]},"sesemi.config.structs.EMAConfig":{decay:[5,3,1,"id32"]},"sesemi.config.structs.LRSchedulerConfig":{frequency:[5,3,1,"id33"],interval:[5,3,1,"id34"],monitor:[5,3,1,"id35"],name:[5,3,1,"id36"],scheduler:[5,3,1,"id37"],strict:[5,3,1,"id38"]},"sesemi.config.structs.LossCallableConfig":{callable:[5,3,1,"id39"],reduction:[5,3,1,"id40"],scale_factor:[5,3,1,"id41"],scheduler:[5,3,1,"id42"]},"sesemi.config.structs.LossHeadConfig":{head:[5,3,1,"id43"],reduction:[5,3,1,"id44"],scale_factor:[5,3,1,"id45"],scheduler:[5,3,1,"id46"]},"sesemi.config.structs.RunConfig":{accelerator:[5,3,1,"id47"],batch_size_per_gpu:[5,3,1,"id48"],data_root:[5,3,1,"id49"],dir:[5,3,1,"id50"],gpus:[5,3,1,"id51"],id:[5,3,1,"id52"],mode:[5,3,1,"id53"],num_epochs:[5,3,1,"id54"],num_iterations:[5,3,1,"id55"],num_nodes:[5,3,1,"id56"],pretrained_checkpoint_path:[5,3,1,"id57"],resume_from_checkpoint:[5,3,1,"id58"],seed:[5,3,1,"id59"]},"sesemi.config.structs.RunMode":{FIT:[5,3,1,"id60"],TEST:[5,3,1,"id61"],VALIDATE:[5,3,1,"id62"]},"sesemi.config.structs.SESEMIBaseConfig":{data:[5,3,1,"id63"],learner:[5,3,1,"id64"],run:[5,3,1,"id65"],trainer:[5,3,1,"id66"]},"sesemi.config.structs.SESEMIInferenceConfig":{batch_size:[5,3,1,"id67"],checkpoint_path:[5,3,1,"id68"],crop_dim:[5,3,1,"id69"],data_dir:[5,3,1,"id70"],ncrops:[5,3,1,"id71"],no_cuda:[5,3,1,"id72"],outfile:[5,3,1,"id73"],oversample:[5,3,1,"id74"],resize:[5,3,1,"id75"],topk:[5,3,1,"id76"],workers:[5,3,1,"id77"]},"sesemi.config.structs.SESEMIPseudoDatasetConfig":{batch_size:[5,3,1,"id78"],checkpoint_path:[5,3,1,"id79"],dataset:[5,3,1,"id80"],gpus:[5,3,1,"id81"],image_getter:[5,3,1,"id82"],num_workers:[5,3,1,"id83"],output_dir:[5,3,1,"id84"],postaugmentation_transform:[5,3,1,"id85"],preprocessing_transform:[5,3,1,"id86"],seed:[5,3,1,"id87"],symlink_images:[5,3,1,"id88"],test_time_augmentation:[5,3,1,"id89"]},"sesemi.datamodules":{SESEMIDataModule:[4,1,1,""]},"sesemi.datamodules.SESEMIDataModule":{test:[4,3,1,""],test_dataloader:[4,2,1,""],train:[4,3,1,""],train_batch_sizes:[4,3,1,""],train_batch_sizes_per_gpu:[4,3,1,""],train_batch_sizes_per_iteration:[4,3,1,""],train_dataloader:[4,2,1,""],val:[4,3,1,""],val_dataloader:[4,2,1,""]},"sesemi.datasets":{ImageFile:[4,1,1,""],PseudoDataset:[4,1,1,""],cifar100:[4,4,1,""],cifar10:[4,4,1,""],concat:[4,4,1,""],dataset:[4,4,1,""],default_is_vaild_file:[4,4,1,""],get_image_files:[4,4,1,""],image_file:[4,4,1,""],image_folder:[4,4,1,""],pseudo:[4,4,1,""],register_dataset:[4,4,1,""],stl10:[4,4,1,""]},"sesemi.learners":{Classifier:[4,1,1,""]},"sesemi.learners.Classifier":{backbone:[4,5,1,""],backbone_ema:[4,5,1,""],compute_validation_outputs:[4,2,1,""],configure_optimizers:[4,2,1,""],forward:[4,2,1,""],head:[4,5,1,""],head_ema:[4,5,1,""],log_validation_metrics:[4,2,1,""],training_epoch_end:[4,2,1,""],training_step:[4,2,1,""],training_step_end:[4,2,1,""],validation_epoch_end:[4,2,1,""],validation_step:[4,2,1,""],validation_step_end:[4,2,1,""]},"sesemi.losses":{kl_div_loss:[4,4,1,""],softmax_mse_loss:[4,4,1,""]},"sesemi.models":{backbones:[7,0,0,"-"],heads:[8,0,0,"-"],utils:[6,0,0,"-"]},"sesemi.models.backbones":{base:[7,0,0,"-"],resnet:[7,0,0,"-"],timm:[7,0,0,"-"]},"sesemi.models.backbones.base":{Backbone:[7,1,1,""]},"sesemi.models.backbones.base.Backbone":{freeze:[7,2,1,""],out_features:[7,3,1,"id0"]},"sesemi.models.backbones.resnet":{CIFARDeepResidualBlock:[7,1,1,""],CIFARResNet:[7,1,1,""],CIFARResidualBlock:[7,1,1,""]},"sesemi.models.backbones.resnet.CIFARDeepResidualBlock":{forward:[7,2,1,""],training:[7,3,1,""]},"sesemi.models.backbones.resnet.CIFARResNet":{forward:[7,2,1,""],out_features:[7,3,1,""],training:[7,3,1,""]},"sesemi.models.backbones.resnet.CIFARResidualBlock":{forward:[7,2,1,""],training:[7,3,1,""]},"sesemi.models.backbones.timm":{PyTorchImageModels:[7,1,1,""]},"sesemi.models.backbones.timm.PyTorchImageModels":{forward:[7,2,1,""],out_features:[7,3,1,""],training:[7,3,1,""]},"sesemi.models.heads":{base:[8,0,0,"-"],loss:[8,0,0,"-"]},"sesemi.models.heads.base":{Head:[8,1,1,""],LinearHead:[8,1,1,""]},"sesemi.models.heads.base.Head":{in_features:[8,3,1,"id0"],out_features:[8,3,1,"id1"]},"sesemi.models.heads.base.LinearHead":{forward:[8,2,1,""],in_features:[8,3,1,""],out_features:[8,3,1,""],training:[8,3,1,""]},"sesemi.models.heads.loss":{ConsistencyLossHead:[8,1,1,""],EMAConsistencyLossHead:[8,1,1,""],EntropyMinimizationLossHead:[8,1,1,""],FixMatchLossHead:[8,1,1,""],JigsawPredictionLossHead:[8,1,1,""],LossHead:[8,1,1,""],RotationPredictionLossHead:[8,1,1,""]},"sesemi.models.heads.loss.ConsistencyLossHead":{forward:[8,2,1,""],training:[8,3,1,""]},"sesemi.models.heads.loss.EMAConsistencyLossHead":{forward:[8,2,1,""],training:[8,3,1,""]},"sesemi.models.heads.loss.EntropyMinimizationLossHead":{forward:[8,2,1,""],training:[8,3,1,""]},"sesemi.models.heads.loss.FixMatchLossHead":{forward:[8,2,1,""],training:[8,3,1,""]},"sesemi.models.heads.loss.JigsawPredictionLossHead":{training:[8,3,1,""]},"sesemi.models.heads.loss.LossHead":{build:[8,2,1,""],forward:[8,2,1,""],logger:[8,3,1,""],training:[8,3,1,""]},"sesemi.models.heads.loss.RotationPredictionLossHead":{build:[8,2,1,""],forward:[8,2,1,""],training:[8,3,1,""]},"sesemi.models.utils":{load_torch_hub_model:[6,4,1,""]},"sesemi.ops":{conf:[10,0,0,"-"],inference:[9,0,0,"-"],pseudo_dataset:[9,0,0,"-"]},"sesemi.ops.inference":{Predictor:[9,1,1,""],predict:[9,4,1,""]},"sesemi.ops.inference.Predictor":{predict:[9,2,1,""]},"sesemi.ops.pseudo_dataset":{TestTimeAugmentationCollator:[9,1,1,""],apply_model_to_test_time_augmentations:[9,4,1,""],default_image_getter:[9,4,1,""],default_test_time_augmentation:[9,4,1,""],pseudo_dataset:[9,4,1,""],task:[9,4,1,""]},"sesemi.schedulers":{lr:[11,0,0,"-"],weight:[11,0,0,"-"]},"sesemi.schedulers.lr":{PolynomialLR:[11,1,1,""]},"sesemi.schedulers.lr.PolynomialLR":{get_lr:[11,2,1,""]},"sesemi.schedulers.weight":{SigmoidRampupScheduler:[11,1,1,""],WeightScheduler:[11,1,1,""]},"sesemi.trainer":{cli:[12,0,0,"-"],conf:[13,0,0,"-"]},"sesemi.trainer.cli":{open_sesemi:[12,4,1,""]},"sesemi.transforms":{GammaCorrection:[4,1,1,""],GaussianBlur:[4,1,1,""],MultiViewTransform:[4,1,1,""],TwoViewsTransform:[4,1,1,""],center_crop_transforms:[4,4,1,""],cifar_test_transforms:[4,4,1,""],cifar_train_transforms:[4,4,1,""],multi_crop_transforms:[4,4,1,""],train_transforms:[4,4,1,""]},"sesemi.utils":{assert_same_classes:[4,4,1,""],compute_device_names:[4,4,1,""],compute_num_gpus:[4,4,1,""],copy_config:[4,4,1,""],load_checkpoint:[4,4,1,""],reduce_tensor:[4,4,1,""],sigmoid_rampup:[4,4,1,""],validate_paths:[4,4,1,""]},sesemi:{collation:[4,0,0,"-"],config:[5,0,0,"-"],datamodules:[4,0,0,"-"],datasets:[4,0,0,"-"],learners:[4,0,0,"-"],losses:[4,0,0,"-"],models:[6,0,0,"-"],ops:[9,0,0,"-"],schedulers:[11,0,0,"-"],trainer:[12,0,0,"-"],transforms:[4,0,0,"-"],utils:[4,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"],"5":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function","5":"py:property"},terms:{"0":[1,3,4,5,7,8,9],"00028":4,"0005":3,"001":3,"01":4,"01780":8,"02":4,"02242":[4,8],"05709":4,"06864":8,"07685":8,"07728":8,"1":[1,3,4,5,9,11],"10":[3,4,7],"100":[3,4],"16":[3,4,5],"1610":[4,8],"1703":8,"1704":4,"18":7,"180":4,"1803":8,"1903":8,"1e":4,"2":[3,4,7],"2001":8,"2002":4,"2004":8,"2016":7,"2019":1,"2020":8,"224":[4,5],"225":4,"229":4,"256":[4,5],"270":4,"3":[1,4],"32":3,"4":[3,7,8],"406":4,"42":[3,5],"456":4,"485":4,"5":[3,4,5,8],"50":3,"6":[1,4,5,8],"770":7,"778":7,"80":1,"9":3,"90":[1,4],"91":1,"96f2b50b5d3613adf9c27049b2a888c7":8,"99":4,"999":5,"case":[1,3,4,9],"class":[0,3,4,5,7,8,9,11],"default":[0,3,4,5,9],"do":4,"enum":5,"final":3,"float":[4,5,7,8,11],"function":[3,4,5,7,8,12],"import":[0,3,4],"int":[3,4,5,7,8,9,11],"long":1,"new":8,"null":[3,5],"return":[3,4,5,6,9],"switch":4,"true":[3,4,5,7,9],"while":[1,4,5,7,8],A:[1,3,4,5,7,8,9,11],And:3,As:[3,8],At:4,But:4,For:[0,3,4,5],If:[1,3,4,5],In:[0,1,3,4],It:4,The:[0,1,3,4,5,6,7,8,9,11,12],Then:1,There:[3,4],These:[0,3,5],To:[1,3],With:[1,4],_:1,_global_:3,_lrschedul:11,_target_:[3,5],ab:[4,8],abl:3,abov:[1,3,4],absolut:5,acc:4,acceler:[3,5],access:[1,3],accuraci:[1,4],across:1,actual:[3,4,8],ad:8,adam:4,adapt:8,add:[0,1,3,4],add_imag:4,addit:[3,4,5,8],addition:[0,3],advanc:[1,3,4],after:[4,5],afterward:[7,8],ahead:3,ai:[1,3],aim:3,alexei:8,algorithm:4,all:[1,3,4,5,7,8],along:[3,4],also:[1,3,4,5],alter:8,although:[3,7,8],amazonaw:[1,3],an:[1,3,4,5,8,9],analysi:7,ani:[1,3,4,5,8,9],annot:3,anyth:[3,4],api:[3,5],append:4,appli:[3,4,5,9],applic:1,apply_model_to_test_time_augment:9,approach:0,appus:1,ar:[0,1,3,4,5,8,9],arbitrari:[0,4],architectur:[1,6],arg:[1,3,4,6],argmax:4,argument:[0,3,4,8],articl:[7,8],arxiv:[4,8],aspect:5,assert_same_class:4,assign:[4,9],associ:4,assum:[3,9],attribut:[3,5],attributeresolv:[3,5],augment:[4,5,9],author:[1,7,8],automat:[0,4],avail:[3,4,5],averag:[4,5],avg:[3,7],avoid:8,back:[3,4],backbon:[1,2,3,4,5,6,8],backbone_ema:[4,8],backend:4,background:3,backprop:4,backward:4,bar:[3,4],bare:3,base:[1,2,3,4,5,6,9,11],basecontain:5,baselin:3,bash:1,batch:[1,3,4,5,8,9],batch_cifar:4,batch_compatible_tensor:9,batch_idx:4,batch_index:4,batch_mnist:4,batch_parts_output:4,batch_sampl:5,batch_siz:[4,5],batch_size_per_gpu:[3,5],bbrattoli:4,becaus:4,been:4,befor:0,behavior:0,being:4,below:[1,3,4],benchmark:1,berthelot:8,best_top1:4,between:4,big:4,bilinear:4,bind:1,bit:4,block:7,blur:4,bone:3,booktitl:1,bool:[4,5,7,8,9],boost:1,both:[1,3],box:[1,3],branch:[1,3],brief:3,browser:1,build:[1,3,4,8],builder:[3,4],built:[0,1],c:[1,3],cach:1,calcul:4,call:[4,7,8],callabl:[3,4,5,9],callback:[3,4,5],can:[0,1,3,4,5,8],cannot:[3,5],care:[7,8],carlini:8,cc:[3,8],cd:[0,1,3],center:[4,5],center_crop_transform:[3,4],certain:[3,5],chang:[3,4],check:[0,3,4],checkpoint:[1,3,4,5],checkpoint_path:[1,4,5],child:9,choic:1,choos:4,chun:8,cifar100:4,cifar10:[4,7],cifar:[4,7],cifar_load:4,cifar_test_transform:4,cifar_train_transform:4,cifardeepresidualblock:7,cifarresidualblock:7,cifarresnet:7,cite:1,ckpt:1,classif:[7,8],classifi:[1,3,4,5],classifierconfig:[3,5],classifierhparam:[3,4,5],classifiermodelconfig:5,cleanli:3,cli:[0,2,3,4,5],clone:3,closur:4,cn:[0,1,3],code:[1,3,4],codebas:3,coeffici:5,colin:8,collat:[1,2,3,5],collate_fn:[3,5],collect:[3,4],com:[1,3,4],come:0,comma:[3,4,5],command:[0,1,9],common:[3,5],compat:9,competit:1,complex:3,compos:[0,3,4],composit:3,comput:[3,4,5,7,8],compute_device_nam:4,compute_num_gpu:4,compute_validation_output:4,concat:4,concaten:9,concept:1,condit:4,conf:[2,4,9,12],confer:7,confid:8,config:[0,2,3,4,9,12],configur:[0,1,4,5,9,12,13],configure_optim:4,consid:1,consist:8,consistencylosshead:8,construct:[0,3],contain:[4,5,6],contemporari:1,content:[2,3],continu:4,contribut:1,control:4,convent:1,copi:[4,5],copy_config:4,core:[3,4,9],correct:4,correspond:[4,9],cosineann:4,could:[4,6],coupl:3,cpu:[3,4,5],creat:[1,3],crop:[4,5],crop_dim:[4,5],crop_patch:4,cross:3,crossentropyloss:3,cubuk:8,cuda:5,curl:[1,3],curr_it:4,current:[1,3,4],custom:[0,3,4,5],cv:1,cvpr:7,cycl:4,d:4,data:[0,1,3,4,5,8,9],data_dir:5,data_root:[3,5],dataclass:3,dataconfig:[3,5],dataload:[4,5],dataloader_i_output:4,dataloader_idx:4,dataloader_out:4,dataloader_output_result:4,dataloaderconfig:[3,5],datamodul:[1,2],dataset:[1,2,5,9],datasetconfig:[3,5],david:8,ddp2:4,ddp:[3,4,5],decai:5,decid:4,decod:4,decor:[3,4],deep:[0,7],deepspe:4,def:[3,4],default_image_gett:9,default_is_vaild_fil:4,default_load:4,default_test_time_augment:9,defin:[0,3,4,7,8],definit:[7,8],degre:4,demonstr:1,denomin:4,depend:1,describ:[4,7],design:7,desir:0,detail:[1,4,9],determin:4,develop:3,deviat:4,devic:[4,9],dict:[3,4,5,8,9],dictconfig:4,dictionari:[3,4,5,8,9],didn:4,differ:[0,3,4,6],dim:4,dimens:[4,5],dir:[3,5],directli:[3,5],directori:[0,1,3,4,5],dis_opt:4,dis_sch:4,disabl:[4,5],displai:4,distanc:4,distribut:4,dive:0,diverg:4,doc:5,docker_buildkit:1,dockerfil:1,document:1,doe:5,dogu:8,don:[1,3,4],done:[1,3],doubl:8,download:[1,3,4,6],downsampl:7,dp:[3,4,5],drop:5,drop_last:[3,5],drop_rat:[3,7],dst:1,dure:[1,3,5,8],dynam:0,e:[1,3,4,5],each:[3,4,5,9],easi:[0,1,3],easili:3,echo:1,edg:5,edit:3,either:[3,4,5,9],ekin:8,element:9,ema:[4,5,8],emaconfig:5,emaconsistencylosshead:8,en:[3,5],enabl:[0,1,3,4],encod:4,end:4,enforc:[4,5],ensur:[1,4],enter:[1,3],entri:4,entropi:[3,8],entropyminimizationlosshead:8,environ:1,epoch:[1,3,4,5],equal:4,eras:4,error:4,essenti:3,etc:[1,3,5],eval:4,evalu:1,everi:[4,7,8],exampl:[0,1,4,5,9],example_imag:4,except:6,exist:[4,5],expand:1,experi:[1,3,4],explor:1,exponenti:[4,5],exponentiallr:4,expos:[3,5],extern:[0,3],extract:[1,5],factor:5,factori:5,fals:[3,4,5,7],familiar:0,fanci:1,fancier:4,fast:[1,3],fastai:1,featur:[3,7,8],few:1,file:[0,3,4,5,8],filesystem:4,fill:3,final_metr:4,final_valu:4,find:[1,3,4],first:[3,4,9],fit:[3,4,5],fixmatch:8,fixmatchlosshead:8,flexibl:[0,3],flip:4,flyreelai:1,folder:[3,4],follow:[1,3,4,5,8],foo:3,form:3,format:[3,4],former:[4,7,8],forward:[3,4,7,8],found:4,freez:[3,7],frequenc:[4,5],from:[0,1,3,4,5,7,8,9],full:[3,4,5],fulli:3,g:[1,3,4,5],gamma:4,gamma_rang:4,gammacorrect:4,gan:4,gaussian:4,gaussianblur:4,gen_opt:4,gen_sch:4,gener:[3,4,5,7,9],get:[4,5],get_image_fil:4,get_lr:11,getter:9,git:1,github:[1,4,6],given:[0,4],global:9,global_pool:[3,7],go:3,goal:1,goe:4,gpu:[1,3,4,5],gpu_0_pr:4,gpu_1_pr:4,gpu_n_pr:4,gradient:4,grayscal:4,green:1,grid:4,grid_siz:4,group:[0,3,5],guid:[1,4],h:3,ha:4,ham:4,han:8,handl:[4,6],happen:4,hardwar:4,have:[1,3,4],he2016deeprl:7,he:7,head:[2,3,4,5,6],head_ema:4,height:4,help:3,here:[1,4],hidden:4,high:[0,1],highlight:5,home:1,hook:[4,7,8],horiziont:4,host:1,how:[3,4],howev:[1,3,4],hparam:[3,4,5],html:[3,5],http:[1,3,4,5,8],hub:6,hydra:[1,3,9,13],hydra_config:9,hydraconfig:9,hyper:1,hyperparamet:[3,5],id:[1,3,4,5,9],idea:8,ident:9,identifi:[3,5,9],idx:4,ieee:7,ignor:[4,7,8],imag:[3,4,5,7,8,9],image_augment:4,image_fil:4,image_fold:[3,4],image_gett:[5,9],image_transform:[3,4,5],imagecla:[1,3],imagefil:4,imagefold:4,imagenet:1,imagenett:1,imagewoof2:[1,3],imagewoof:3,imagewoof_rot:[1,3],implement:[4,8],improv:[1,4],in_channel:7,in_featur:8,includ:4,incorpor:1,increas:4,index:[1,4],indic:5,individu:4,infer:[2,4,5],info:3,inform:3,initi:[3,5],inject:0,inner:4,inproceed:1,input:[3,4,7,8,9],input_backbon:[3,8],input_data:[3,8],insid:4,inspect:3,instal:3,instanc:[3,4,7,8],instanti:[0,3,5],instead:[1,3,5,7,8],instruct:1,integ:[3,4,5],integr:1,interchang:0,interest:4,interfac:[3,7,8,11],intern:[0,3],interpol:[3,4],interpolaton:4,interv:[4,5],invok:5,involv:0,io:[3,5],ipc:1,is_valid_fil:4,item:4,iter:[3,4,5],iterabledataset:[3,4],iterations_per_epoch:[3,5],iters_per_epoch:[3,11],ith:4,its:[4,9],jian:7,jigsaw:[4,8],jigsawpredictionlosshead:8,jigsawpuzzlepytorch:4,jigsawtransform:4,journal:[7,8],just:3,kaim:7,kei:[0,4],keyword:4,kihyuk:8,kind:[3,5],kl_div_loss:4,know:[1,4],known:3,kullback:4,kurakin:8,kwarg:[3,4,6,7,8],label:[1,5,9],labels_hat:4,larg:[1,4],last:[1,4,5],last_epoch:11,later:4,latest:[1,3,5],latter:[4,7,8],lbfg:4,leaderboard:1,learn:[1,4,5,7,8,11],learner:[1,2,3,5],learnerconfig:[3,5],learningratemonitor:4,least:1,leibler:4,len:4,let:1,level:[0,4],li:8,liang:8,librari:[1,3],light:1,lightn:[1,3,4,5,8],lightning_log:1,lightningdatamodul:4,lightningloggerbas:8,lightningmodul:4,like:[1,4],limit:1,line:9,linear_featur:7,linearhead:[4,8],list:[0,1,3,4,5,9],littl:1,ll:3,load:[3,4,5,6],load_checkpoint:4,load_torch_hub_model:6,loader:[3,4,5],loader_a:4,loader_b:4,loader_n:4,local:3,locat:4,log:[1,3,4,5],log_dict:4,log_validation_metr:4,logdir:1,logger:[4,8],logit:9,look:[0,3],lookup:5,loss:[1,2,3,5,6],loss_fn:8,losscallableconfig:5,losshead:8,lossheadconfig:5,lowercas:4,lr:[2,3,4],lr_dict:4,lr_pow:[3,11],lr_schedul:[3,4,5,11],lrschedulerconfig:5,lstm:4,machin:1,made:4,mai:[3,8],main:[0,1,3,4,5,12],maintain:5,make:[0,3],make_grid:4,manag:[0,3],mani:[1,4],map:[3,5],match:4,max:3,max_it:[3,11],max_iter:[3,5],max_retri:6,maxim:4,maximum:[3,5,6],mean:[3,4,5,8],mechan:3,mention:4,metadata:9,method:[1,4,5],metric:[3,4,5],metric_to_track:4,metric_v:4,might:4,mini:5,miniconda:1,minim:8,miss:3,mkdir:1,ml:1,mnist:4,mnist_load:4,mode:[1,3,4,5],model:[1,2,3,4,5,9],model_di:4,model_gen:4,model_path:9,modelcheckpoint:3,modern:1,modif:3,modul:[1,2,3],moduledict:8,momentum:3,monitor:[1,3,4,5],more:[1,4,5],most:[1,4,5],mount:1,move:5,mse:8,multi:[4,5],multi_crop_transform:4,multi_gpu:4,multipl:[0,4,5],multiviewtransform:4,must:4,mutablemap:5,n:[4,7],n_critic:4,name:[1,3,4,5,6,7],nce:4,nce_loss:4,ncrop:[5,9],necessari:[1,4],need:[1,3,4,7,8],nest:3,nesterov:3,network:7,neurip:1,next:[3,4],nichola:8,nip:8,nn:[3,4,7,8,9],no_cuda:5,node:[3,5],noisi:1,none:[3,4,5,7,8,11],norm:4,normal:4,note:[1,3],now:4,num_class:[3,5],num_crop:4,num_epoch:[3,5],num_gpu:[3,4,5],num_iter:[3,5],num_nod:[3,5],num_pretext_class:8,num_residual_block:7,num_task:9,num_view:4,num_work:[3,5],number:[3,4,5,6,7,8,9],nutshel:0,nvidia:1,object:[0,3,4,5,8,9,11],obtain:1,often:4,omegaconf:[3,4,5],one:[1,3,4,5,7,8],onli:[1,4],op:[2,4],open:1,open_sesemi:[0,1,3,12],oper:[4,5,9],optim:[3,4,5,11],optimizer_idx:4,optimizer_on:4,optimizer_step:4,optimizer_two:4,option:[1,3,4,5,7,8,9,11],order:[3,4],org:[1,4,5,8],origin:9,os:1,other:[3,4,8],otherwis:3,our:1,out:[1,3,4],out_channel:7,out_featur:[7,8],outer:4,outfil:5,output:[3,4,5,7,8],output_dir:5,over:4,overrid:[3,4],overridden:[4,7,8],oversampl:5,overview:0,own:[0,3,4],p:4,p_blur:4,p_eras:4,p_grayscal:4,p_hflip:4,packag:[0,1,2,3],page:[1,3,4,7],paper:[1,7,8],paradigm:1,paramet:[1,3,4,5,6,7,8,9,12],paramref:4,parent:[3,5],pars:3,part:4,particular:3,pass:[3,4,5,7,8],patch:4,path:[0,3,4,5],pattern:[4,7],pdf:8,per:[3,4,5],perform:[1,5,7,8],permut:4,phi:1,pi:8,pil:9,pin:5,pin_memori:[3,5],pip:3,placehold:8,pleas:[1,3,4],point:3,polynomi:11,polynomiallr:[3,11],popular:1,portion:4,posit:4,possibl:[3,4],postaugmentation_transform:[5,9],postiv:4,power:3,practition:1,precis:4,pred:4,predict:[3,4,5,8,9],predict_head:8,predictor:9,prefer:1,prefix:4,prepar:[4,6],prepare_data:4,preprocess:5,preprocessing_transform:[5,9],present:4,pretrain:[1,3,5,7],pretrained_checkpoint_path:[1,3,5],previou:4,primer:[1,3],principl:1,probabl:4,procedur:4,process:[1,4,5],produc:4,progress:4,propag:4,properti:4,protocol:1,provid:[0,3],pseudo:[4,5,9],pseudo_dataset:[2,4],pseudocod:4,pseudodataset:4,put:4,pwd:1,pypi:1,python:[0,1,3],pytorch:[1,3,4,5,7,8],pytorch_lightn:[3,4,8],pytorchimagemodel:[3,7],quickstart:1,raffel:8,rais:6,ramp:[4,11],rampup:4,rampup_it:4,random:[3,4,5],random_resized_crop:4,randomli:4,rang:4,rate:[1,4,5,11],rather:5,ratio:5,re:1,readthedoc:[3,5],realist:1,reason:3,recip:[7,8],recognit:7,recommend:4,recurs:4,reduc:4,reduce_tensor:4,reducelronplateau:4,reduct:[3,4,5],refer:[3,5],referenc:[3,5,7],regist:[3,4,7,8],register_dataset:[3,4],registri:[4,5],regular:3,regularization_loss_head:[3,5],rel:[3,5],relat:1,relev:1,reload:4,reload_dataloaders_every_n_epoch:4,ren:7,repo:[1,6],repositori:[1,3,6,7],requir:[1,4],residu:7,resiz:[4,5],resnet50d:[3,7],resnet:[2,4,6],resolut:5,resolv:[2,3,4],respect:5,restor:[3,5],result:4,resume_from_checkpoint:[3,5],return_supervised_label:4,rich:1,rm:1,robust:1,root:[1,3,4,5],rotat:[3,4,8],rotation_predict:3,rotationpredictionlosshead:[3,8],rotationtransform:[3,4],run:[0,1,3,4,5,7,8,9],runconfig:[3,5],runmod:[3,5],runtim:[0,3],rwightman:7,s3:[1,3],s:[1,3,4,5,7,9,12],same:[3,4],sampl:[3,4],sample_img:4,sampler:[4,5],save:[5,8],save_last:3,save_top_k:3,scale:[4,5],scale_factor:[3,5],schedul:[2,3,4,5],schema:0,scmode:4,search:[0,1,3,4],second:4,section:[0,1,3,7],see:[1,3,4,5,7],seed:[3,5],select:[0,4],self:4,semi:[1,8],separ:[3,4,5],sequenc:4,sequenti:4,sesemi:[0,3],sesemi_config:3,sesemi_imag:1,sesemibaseconfig:[3,5,12],sesemiconfigattribut:[3,5],sesemidatamodul:4,sesemiinferenceconfig:[5,9],sesemipseudodatasetconfig:[5,9],set:[0,1,3,4,5],setup:4,sgd:[3,4],shaoq:7,share:8,shot:1,should:[1,3,4,5,7,8],shown:[3,4],shuffl:[3,4,5],sigma_rang:4,sigmoid:[4,11],sigmoid_rampup:4,sigmoidrampupschedul:11,silent:[7,8],simclr:4,similar:[3,4,8],simpli:[1,4],simplifi:8,sinc:[7,8],singl:4,size:[1,3,4,5,9],skip:4,smaller:5,smooth:4,so:4,softmax:4,softmax_mse_loss:4,sohn2020fixmatchss:8,sohn:8,some:[0,3,4],someth:4,somewhat:3,somewher:3,sourc:[1,3,5],special:7,specif:[0,3,4,5],specifi:[3,4,5],split:4,split_batches_for_dp:4,squar:4,src:[1,3],standard:[1,4,9],star:1,start:3,state:[3,4,5],statist:1,step:[4,5,8],still:4,stl10:4,stop:4,stop_rampup:11,store:[3,5],str:[3,4,5,6,7,8,9],strict:[4,5],string:[5,9],struct:[2,3,4,9,12],structur:[0,1,5],structured_config_mod:4,student_backbon:8,student_head:8,sub_batch:4,subclass:[7,8],subdir:5,subet:4,subject:3,submodul:[1,2],subpackag:[1,2],subset:[3,4,5,9],sum:[4,5],sun:7,supervis:[3,4,8],supervised_ema:8,supervised_loss:[3,5],support:[0,1,3,4,5],symlink:5,symlink_imag:5,syntax:[3,5],system:[0,3],t:[1,3,4,9],t_co:4,t_max:4,take:[1,5,7,8],taken:5,tar:[1,3],target:4,target_transform:4,task:[1,4,9],task_id:9,teacher:8,teacher_backbon:8,teacher_head:8,tell:4,tensor:[4,5,8,9],tensorboard:1,tensorflow:1,test:[3,4,5,9],test_dataload:4,test_step:4,test_time_augment:[5,9],testtimeaugmentationcol:9,text:[3,4],tgz:[1,3],than:5,them:[1,4,7,8],thi:[0,1,3,4,5,7,8],thing:4,those:[0,3,4],threshold:8,through:[0,1,3,4],thu:4,time:[1,3,4,5,9],timeout:5,timm:[1,2,4,6],titl:[1,7,8],too:0,tool:[1,9],top1:[3,4],top:4,topk:[5,9],torch:[3,4,5,6,7,8,9,11],torchvis:[3,4],totensor:4,tpu:4,track:[3,8],train:[1,3,4,5,7,8,12,13],train_batch:4,train_batch_s:4,train_batch_sizes_per_gpu:4,train_batch_sizes_per_iter:4,train_data:4,train_dataload:4,train_out:4,train_transform:[3,4],trainer:[2,3,4,5],training_epoch_end:4,training_step:4,training_step_end:4,training_step_output:4,tran:1,transesemi:1,transform:[1,2,3,5],tri:6,trick:1,truncat:4,truncated_bptt_step:4,tune:1,tupl:[4,9],turn:3,two:[3,4],twoviewstransform:4,type:[0,1,3,4,5,7,8],typic:0,u:1,under:[3,4],underli:[0,3,4],unevenli:5,union:[3,4],unit:4,unlabel:[1,4],unless:4,unsupervis:3,up:[4,11],updat:[4,5],us:[0,1,3,4,5,9],usag:1,use_probability_target:4,user:[0,1,3],user_id:1,util:[1,2,5],val:[3,4,5],val_acc:4,val_batch:4,val_data:4,val_dataload:4,val_loss:4,val_out:4,val_step_output:4,valid:[1,3,4,5],validate_path:4,validation_epoch_end:4,validation_step:4,validation_step_end:4,valu:[3,4,5],variabl:3,variou:1,veri:1,version:[1,3,5],version_0:1,via:0,view:[1,3,4],virtual:1,vision:7,volum:8,vu:1,w:1,wai:[0,3],want:3,warmup_epoch:[3,11],warmup_it:11,warmup_lr:[3,11],warn:4,wasserstein:4,we:[3,4],weheth:5,weight:[1,2,3,4,5],weight_decai:3,weightschedul:11,welcom:1,well:[0,3],what:4,whatev:4,when:[3,4,5],where:4,wherea:4,whether:[4,5,9],which:[0,1,3,4,5,8,9],whose:4,why:3,width:4,wish:4,within:[7,8],without:[1,3,4],won:4,work:[0,1,3],worker:5,worker_init_fn:5,workshop:1,would:1,write:5,written:1,www:1,x:[4,7,9],xzv:[1,3],y:4,yaml:[0,3],year:[1,7,8],you:[1,3,4],your:[1,3,4],yourself:4,z:4,zhang:[7,8],zizhao:8},titles:["A Primer on Hydra","Image Classification with Self-Supervised Regularization","sesemi","Quickstart","sesemi package","sesemi.config package","sesemi.models package","sesemi.models.backbones package","sesemi.models.heads package","sesemi.ops package","sesemi.ops.conf package","sesemi.schedulers package","sesemi.trainer package","sesemi.trainer.conf package"],titleterms:{A:0,applic:0,backbon:7,base:[7,8],built:3,citat:1,classif:1,cli:12,collat:4,concept:0,conf:[10,13],config:5,configur:3,content:[1,4,5,6,7,8,9,10,11,12,13],datamodul:4,dataset:[3,4],docker:1,exampl:3,featur:1,get:1,head:8,highlight:1,hydra:0,imag:1,indic:1,infer:9,instal:1,learner:4,loss:[4,8],lr:11,model:[6,7,8],modul:[4,5,6,7,8,9,10,11,12,13],op:[9,10],packag:[4,5,6,7,8,9,10,11,12,13],pip:1,primer:0,pseudo_dataset:9,quickstart:3,regular:1,resnet:7,resolv:5,schedul:11,self:1,sesemi:[1,2,4,5,6,7,8,9,10,11,12,13],start:1,struct:5,structur:3,submodul:[4,5,6,7,8,9,11,12],subpackag:[4,6,9,12],supervis:1,tabl:1,timm:7,tool:[],trainer:[12,13],transform:4,usag:3,util:[4,6],weight:11,why:1}})