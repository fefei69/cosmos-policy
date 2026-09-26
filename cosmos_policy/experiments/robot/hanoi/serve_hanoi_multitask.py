"""HTTP policy server for hanoi_multitask_v6 exports, after the official ALOHA ``deploy.py``.

Request (JSON, ``json_numpy`` encoded like the ALOHA server, or lists):
``{"observation/image": (224, 224, 3) uint8, "observation/state": (7,) float32,
"prompt": <one of the six task prompts, verbatim>}``. The prompt is required on
every request; there is no default task, and a request without a trained
prompt is refused with HTTP 400. Reply: ``{"actions": (16, 4) float32 absolute
XYZ + jaw intent 0/1, "reference_rate_hz": 10, "execution_prefix": 8,
"action_horizon": 16, "task": direction, "task_index", "goal_peg",
"cosmos_hanoi": identity}``. ``GET /identity`` returns the identity alone,
including the six prompts, so the client can compare the contract before
driving the arm. Server dependencies (``fastapi``, ``uvicorn``, ``json_numpy``)
are deployment-side extras; this module is not exercised on the cluster.
"""
import argparse
import logging
import time
import traceback

import numpy as np

from cosmos_policy.experiments.robot.hanoi.multitask_policy import HanoiMultitaskInferenceConfig, HanoiMultitaskPolicy


def build_app(policy):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import json_numpy
    json_numpy.patch()
    app = FastAPI()

    @app.get('/identity')
    def identity():
        return JSONResponse(policy.identity)

    @app.post('/act')
    def act(payload: dict):
        try:
            if 'encoded' in payload:
                import json
                payload = json.loads(payload['encoded'])
            if 'prompt' not in payload:
                return JSONResponse({'error': 'A multitask request must carry one of the six task prompts verbatim'}, status_code=400)
            observation = {
                'observation/image': np.asarray(payload['observation/image'], dtype=np.uint8),
                'observation/state': np.asarray(payload['observation/state'], dtype=np.float32),
                'prompt': payload['prompt'],
            }
            started = time.time()
            reply = policy.infer(observation, seed=int(payload.get('seed', 1)))
            reply['actions'] = reply['actions'].astype(np.float32)
            reply['inference_seconds'] = time.time() - started
            reply.update(policy.identity)
            return JSONResponse(json_numpy.dumps(reply))
        except Exception:  # noqa: BLE001
            logging.error(traceback.format_exc())
            return JSONResponse({'error': traceback.format_exc()}, status_code=400)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='exports/iter_*.pt inside its run directory')
    parser.add_argument('--stats', required=True, help='data/hanoi_cosmos/multitask_v6/dataset_statistics.json')
    parser.add_argument('--embeddings', default='data/hanoi_cosmos/t5_embeddings_multitask.pkl')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8777)
    args = parser.parse_args()
    import os
    os.environ['COSMOS_POLICY_PLATFORM'] = 'hanoi_dense'
    os.environ['HANOI_DENSE_HORIZON'] = '16'
    import uvicorn
    policy = HanoiMultitaskPolicy(HanoiMultitaskInferenceConfig(args.checkpoint, args.stats, args.embeddings,
                                                                num_denoising_steps_action=args.steps))
    logging.info('identity: %s', policy.identity)
    uvicorn.run(build_app(policy), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
