import { ethers } from "hardhat";

async function main() {
  
  const Dao = await ethers.getContractFactory("umaInteract");
  //"0x2Ff10419Fd92498b1143a96Aa0374E9d247B7CBA" mumbai
  const dao = await Dao.deploy()
  await dao.registerDAO("0x7FC5f4f5bE07b698365DA975218909195404eF89",10,0)

  await dao.submitDeployment("testr","0x7FC5f4f5bE07b698365DA975218909195404eF89")
  console.log(await dao.propDetails("testr"))
  await dao.requestData("testr");
  console.log(dao.address)
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
