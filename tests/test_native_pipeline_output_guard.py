import mlx.core as mx
from omlx.cluster.native_text_pipeline import NativeTextPipelineMixin
from omlx.cluster.runtime_optimizations import _supports_coordinator_sampling

class DelegatingModel(NativeTextPipelineMixin):
    def __call__(self,h):
        return self.forward_pipeline(h)

class ExtraCollectiveModel(NativeTextPipelineMixin):
    def __call__(self,h):
        mx.distributed.all_gather(h)
        return self.forward_pipeline(h)

class UnverifiedDelegateModel(NativeTextPipelineMixin):
    def __call__(self,h):
        return self.forward_pipeline(h)
    def forward_pipeline(self,h):
        return h

def test_exact_native_delegate_preserves_coordinator_contract():
    supported,_=_supports_coordinator_sampling(DelegatingModel(),batchable=True,world_size=2)
    assert supported

def test_wrapper_collective_is_rejected():
    supported,_=_supports_coordinator_sampling(ExtraCollectiveModel(),batchable=True,world_size=2)
    assert not supported

def test_replaced_delegate_is_rejected():
    supported,_=_supports_coordinator_sampling(UnverifiedDelegateModel(),batchable=True,world_size=2)
    assert not supported
