pragma solidity ^0.8.0;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {IAToken} from "./IAToken.sol";
import {ILendingPool} from "./ILendingPool.sol";
import {ILendingPoolAddressesProvider} from "./ILendingPoolAddressesProvider.sol";

contract PolicyPool {
    address public owner;
    ILendingPool public lendingPool;
    IERC20 public underlying;
    IAToken public aToken;

    constructor() {
        owner = msg.sender;
        ILendingPoolAddressesProvider provider = ILendingPoolAddressesProvider(0xB53C1a33016B2DC2fF3653530bfF1848a515c8c5);
        lendingPool = ILendingPool(provider.getLendingPool());
        underlying = IERC20(provider.getLendingPoolCore());
        aToken = IAToken(provider.getLendingPoolAToken());
    }

    function deposit(uint256 amount) external {
        underlying.transferFrom(msg.sender, address(this), amount);
        underlying.approve(address(lendingPool), amount);
        lendingPool.deposit(address(underlying), amount, 0);
        aToken.transfer(msg.sender, amount);
    }

    function withdraw(uint256 amount) external {
        aToken.transferFrom(msg.sender, address(this), amount);
        lendingPool.withdraw(address(underlying), amount, address(this));
        underlying.transfer(msg.sender, amount);
    }

    function getPoolBalance() public view returns (uint256) {
        return aToken.balanceOf(address(this));
    }
}
