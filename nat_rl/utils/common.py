import numpy as np

from habitat_sim.utils import viz_utils as vut


def count_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    param_count = sum([np.prod(p.size()) for p in model_parameters])
    return param_count


def run_single_episode(env, policy, make_video=False, eval_train=True):
    obs = env.reset()
    success = False
    episode_id = env.habitat_env.current_episode.episode_id
    if make_video:
        eval_mode = 'eval_train' if eval_train else 'eval_test'
        video_file_path = f'visuals/{eval_mode}/episodeid={episode_id}.mp4'
        video_writer = vut.get_fast_video_writer(video_file_path, fps=30)
        video_writer.append_data(obs['robot_third_rgb'])
    
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        #actions.append(action)
        obs, rew, done, info = env.step(action)

        if make_video:
            video_writer.append_data(obs['robot_third_rgb'])

        if done:
            if env.episode_success():
                success = True

            break
        
    if make_video:
        video_writer.close()
    
    return success


def evaluate_success_rate(env, policy, num_eps=10, make_video=False, eval_train=True):
    num_success = 0
    for _ in range(num_eps):
        success = run_single_episode(env, policy, make_video, eval_train)
        if success:
            num_success += 1
    
    success_rate = num_success / num_eps
    return success_rate