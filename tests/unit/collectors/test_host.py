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


def _linux(os_release: object) -> HostFacts:
    def read() -> dict[str, str]:
        if isinstance(os_release, Exception):
            raise os_release
        return os_release  # type: ignore[return-value]

    return collect_host(
        system=lambda: "Linux", machine=lambda: "x86_64", release=lambda: "7.0", os_release=read
    )


def test_distribution_comes_from_pretty_name() -> None:
    facts = _linux({"PRETTY_NAME": "Ubuntu 26.04.1 LTS", "NAME": "Ubuntu", "VERSION_ID": "26.04"})
    assert facts.distribution == "Ubuntu 26.04.1 LTS"


def test_distribution_falls_back_to_name_and_version() -> None:
    assert _linux({"NAME": "Arch Linux"}).distribution == "Arch Linux"
    assert _linux({"NAME": "Fedora Linux", "VERSION_ID": "44"}).distribution == "Fedora Linux 44"


def test_missing_os_release_leaves_the_distribution_unknown() -> None:
    assert _linux(FileNotFoundError("no os-release")).distribution is None
    assert _linux({}).distribution is None


def test_os_release_is_not_read_on_other_systems() -> None:
    def must_not_be_called() -> dict[str, str]:
        raise AssertionError("os-release read on Windows")

    facts = collect_host(
        system=lambda: "Windows",
        machine=lambda: "AMD64",
        release=lambda: "11",
        os_release=must_not_be_called,
    )
    assert facts.distribution is None
