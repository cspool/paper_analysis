# *C. Threat Model*

We assume a software-based remote attacker confined to a virtual machine, with no access to the hypervisor or the server firmware. The attacker is fully aware of Sigries and its configuration parameters. They can create software capable of issuing DRAM commands to arbitrary rows in any order, provided they adhere to DDR5 specifications. The attacker does not have direct access to Sigries's counter values but is assumed to be capable of inferring when REF or DRFM commands are issued through side-channels [\[17\]](#page-13-3), [\[27\]](#page-13-4).

