from gpu_stack_inspector.collectors.host import collect_host
from gpu_stack_inspector.models import HostFacts


def host(
    system: str = "Linux", machine: str = "x86_64", release: str = "7.0.0-42-generic"
) -> HostFacts:
    return collect_host(system=lambda: system, machine=lambda: machine, release=lambda: release)


def test_reads_system_machine_and_kernel() -> None:
    facts = host()
    assert (facts.system, facts.machine, facts.kernel) == ("Linux", "x86_64", "7.0.0-42-generic")
    assert facts.wsl is False


def test_recognises_a_wsl_kernel() -> None:
    assert host(release="5.15.167.4-microsoft-standard-WSL2").wsl is True


def test_empty_platform_answers_become_unknown() -> None:
    facts = host(system="", machine="", release="")
    assert (facts.system, facts.machine, facts.kernel) == ("unknown", "unknown", None)


def test_real_platform_can_be_read() -> None:
    facts = collect_host()
    assert facts.system
    assert facts.machine
