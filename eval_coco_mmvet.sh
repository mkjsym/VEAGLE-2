#bash bench.sh /checkpoint_path image_token_strategy num_image_tokens

# bash cococap_eagle.sh /data/youngmin/checkpoints/aircache_20token_40epoch/last 6 20
# bash mmvet_eagle.sh /data/youngmin/checkpoints/aircache_20token_40epoch/last 6 20
# bash cococap_eagle.sh /data/youngmin/checkpoints/aircache_100token_40epoch/last 6 20
# bash mmvet_eagle.sh /data/youngmin/checkpoints/aircache_100token_40epoch/last 6 20

# bash cococap_eagle.sh /data/youngmin/checkpoints/cls_20token_40epoch/last 5 20
# bash mmvet_eagle.sh /data/youngmin/checkpoints/cls_20token_40epoch/last 5 20
# bash cococap_eagle.sh /data/youngmin/checkpoints/cls_100token_40epoch/last 5 20
# bash mmvet_eagle.sh /data/youngmin/checkpoints/cls_100token_40epoch/last 5 20

bash cococap_eagle.sh /home/youngmin/workspace/finetune_w_img_1e-4_cls_hidden_llavashare_40epoch_embd/state_40 6 20
bash mmvet_eagle.sh /home/youngmin/workspace/finetune_w_img_1e-4_cls_hidden_llavashare_40epoch_embd/state_40 6 20

#Image Token Strategy
#0:nothing
#1:remove
#2:pool
#3:remove_except_last
#4:remove_image_token_except_first
#5:keep_topk_image_token
#6:keep_topk_image_token_aircache
