from python_scripts.WaveGrad.WaveGrad_policy import WaveGradAgent


class WaveGradTaiAgent(WaveGradAgent):
    def __init__(self, node_num, env_information=None, trajectory_len=20):
        super().__init__(
            node_num=node_num,
            env_information=env_information,
            trajectory_len=trajectory_len,
            update_epochs=5,
            max_grad_norm=0.5,
        )

    def store_transition_tai(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        value,
        success_flag=False,
        safety_penalty=0.0,
    ):
        self.store_transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            value=value,
            success_flag=success_flag,
            safety_penalty=safety_penalty,
        )
