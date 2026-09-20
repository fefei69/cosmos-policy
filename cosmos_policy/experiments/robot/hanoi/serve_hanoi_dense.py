"""HTTP policy server for hanoi_dense_v5 exports, after the official ALOHA ``deploy.py``.

Request (JSON, ``json_numpy`` encoded like the ALOHA server, or lists):
``{"observation/image": (224, 224, 3) uint8, "observation/state": (7,) float32,
"prompt": optional}``. Reply: ``{"actions": (H, 4) float32 absolute XYZ + jaw
intent 0/1, "reference_rate_hz": 10, "execution_prefix": 8, "action_horizon": H,
"cosmos_hanoi": identity}`` where H (16, or 32 for the comparison run) comes from
the run's ``joint_contract.json``. ``GET /identity`` returns the identity alone so the client can
compare the contract before driving the arm. Server dependencies
(``fastapi``, ``uvicorn``, ``json_numpy``) are deployment-side extras; this
module is not exercised on the cluster.
"""
import argparse
import logging
import time
import traceback

import numpy as np

from cosmos_policy.experiments.robot.hanoi.dense_policy import HanoiDenseInferenceConfig, HanoiDensePolicy


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
            observation = {
                'observation/image': np.asarray(payload['observation/image'], dtype=np.uint8),
                'observation/state': np.asarray(payload['observation/state'], dtype=np.float32),
            }
            if 'prompt' in payload:
                observation['prompt'] = payload['prompt']
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
    parser.add_argument('--stats', required=True, help='data/hanoi_cosmos/dense_v5/dataset_statistics.json')
    parser.add_argument('--embeddings', default='data/hanoi_cosmos/t5_embeddings.pkl')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8777)
    args = parser.parse_args()
    import json
    import os
    from pathlib import Path
    run = Path(args.checkpoint).resolve().parent.parent  # run/exports/iter_*.pt
    horizon = int(json.loads((run / 'joint_contract.json').read_text()).get('horizon', 16))
    os.environ['COSMOS_POLICY_PLATFORM'] = 'hanoi_dense'
    os.environ['HANOI_DENSE_HORIZON'] = str(horizon)
    import uvicorn
    policy = HanoiDensePolicy(HanoiDenseInferenceConfig(args.checkpoint, args.stats, args.embeddings,
                                                        num_denoising_steps_action=args.steps, chunk_size=horizon))
    logging.info('identity: %s', policy.identity)
    uvicorn.run(build_app(policy), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
