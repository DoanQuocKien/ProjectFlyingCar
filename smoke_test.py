import json,sys,torch

nb=json.load(open('main.ipynb', 'r', encoding='utf-8'))
src=None
for c in nb['cells']:
    s=''.join(c.get('source',[]))
    if 'class TransferMobileNetSSD' in s or 'class MobileNet' in s and 'SSD' in s:
        src=s
        break
if not src:
    for c in nb['cells']:
        s=''.join(c.get('source',[]))
        if 'class TransferMobileNetSSD' in s:
            src=s; break
if not src:
    print('MODEL_CELL_NOT_FOUND')
    sys.exit(2)

# avoid downloading pretrained weights during smoke test
src=src.replace('pretrained=True','pretrained=False')

# execute model cell
gl={}
try:
    exec(src, gl)
except Exception as e:
    print('MODEL_CELL_EXEC_ERROR', e)
    sys.exit(3)

# find class
Model=None
for k,v in gl.items():
    if isinstance(v, type) and v.__name__ in ('TransferMobileNetSSD','TransferMobileNetv3SSD','MobileNetV3SSD'):
        Model=v; break
if Model is None:
    if 'TransferMobileNetSSD' in gl:
        Model=gl['TransferMobileNetSSD']
if Model is None:
    for k,v in gl.items():
        if isinstance(v,type) and 'MobileNet' in v.__name__:
            Model=v; break
if Model is None:
    print('MODEL_CLASS_NOT_FOUND')
    sys.exit(4)

# instantiate safely
try:
    kwargs={}
    import inspect
    params=inspect.signature(Model.__init__).parameters
    if 'num_classes' in params:
        kwargs['num_classes']=5
    if 'image_size' in params:
        kwargs['image_size']=320
    if 'grid_size' in params:
        kwargs['grid_size']=10
    if 'pretrained' in params:
        kwargs['pretrained']=False
    m=Model(**kwargs)
except Exception as e:
    try:
        m=Model(5)
    except Exception as e2:
        print('MODEL_INSTANTIATE_ERROR', e, e2)
        sys.exit(5)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m.to(device)
m.train()

# small synthetic batch
x=torch.randn(2,3,320,320,device=device)
try:
    out=m(x)
except Exception as e:
    print('FORWARD_ERROR', e)
    sys.exit(6)

# compute simple loss and backward
try:
    if isinstance(out,(list,tuple)):
        loss=0
        for o in out:
            try:
                loss = loss + o.sum()
            except:
                pass
    else:
        loss = out.sum()
    loss.backward()
except Exception as e:
    print('BACKWARD_ERROR', e)
    sys.exit(7)

print('SMOKE_OK')
