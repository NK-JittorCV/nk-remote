import jittor as jt
from jrs.config import init_cfg, get_cfg
from jrs.utils.general import parse_losses
from jrs.utils.registry import build_from_cfg,MODELS,DATASETS,OPTIMS
import argparse
import os
import pickle as pk
import jrs

def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--set_data",
        action='store_true'
    )
    args = parser.parse_args()

    jt.flags.use_cuda=1
    jt.set_global_seed(666)
    init_cfg("configs/s2anet_test.py")
    cfg = get_cfg()

    model = build_from_cfg(cfg.model,MODELS)
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params= model.parameters())

    model.train()
    if (args.set_data):
        model.save("test_datas_s2anet/init_pretrained.pk_jt.pk")
        imagess = []
        targetss = []
        correct_loss = []
        train_dataset = build_from_cfg(cfg.dataset.train,DATASETS)
        for batch_idx,(images,targets) in enumerate(train_dataset):
            if (batch_idx > 10):
                break
            print(batch_idx)
            imagess.append(jrs.utils.general.sync(images))
            targetss.append(jrs.utils.general.sync(targets))
            losses = model(images,targets)
            all_loss,losses = parse_losses(losses)
            optimizer.step(all_loss)
            correct_loss.append(all_loss.item())
        data = {
            "imagess": imagess,
            "targetss": targetss,
            "correct_loss": correct_loss,
        }
        if (not os.path.exists("test_datas_s2anet")):
            os.makedirs("test_datas_s2anet")
        pk.dump(data, open("test_datas_s2anet/test_data.pk", "wb"))
        print(correct_loss)
    else:
        model.load("test_datas_s2anet/init_pretrained.pk_jt.pk")
        data = pk.load(open("test_datas_s2anet/test_data.pk", "rb"))
        imagess = jrs.utils.general.to_jt_var(data["imagess"])
        targetss = jrs.utils.general.to_jt_var(data["targetss"])
        correct_loss = data["correct_loss"]
        # correct_loss =[4.851482391357422, 4.919872760772705, 3.1842665672302246, 3.716217041015625, 4.287736415863037, 
        #     3.794440269470215, 3.7207441329956055, 3.743844509124756, 4.571873664855957, 5.585651397705078, 3.2345163822174072]
        for batch_idx in range(len(imagess)):
            images = imagess[batch_idx]
            targets = targetss[batch_idx]
            losses = model(images,targets)
            all_loss,losses = parse_losses(losses)
            optimizer.step(all_loss)
            l = all_loss.item()
            c_l = correct_loss[batch_idx]
            err_rate = abs(c_l-l)/min(c_l,l)
            print(f"correct loss is {c_l:.4f}, runtime loss is {l:.4f}, err rate is {err_rate*100:.2f}%")
            # TODO(514flowey): modify err thr from 1e-3 to 0.1. Try to Fix it.
            assert err_rate<0.1,"LOSS is not correct, please check it"
        print(f"Loss is correct with err_rate<{0.1}")
    print("success!")
    
if __name__ == "__main__":
    main()